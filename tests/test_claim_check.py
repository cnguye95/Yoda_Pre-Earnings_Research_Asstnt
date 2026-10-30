"""Offline tests for the claim check (yoda/report/claim_check.py): verdicts, second opinions, repairs, removals.

No model is called: structured_call is replaced by a fake that fails the
claims each test names.

Run with: python -m pytest -q
"""

import re

import pytest

from samples import claim, sample_report, sample_store
from yoda.modes.llm import NoStructuredOutput, StructuredReply
from yoda.report import claim_check
from yoda.report.claim_check import (
    CHECK_MODEL, SECOND_MODEL, Repair, Repairs, Verdict, Verdicts, apply_repairs, check_claims, remove_claims,
)
from yoda.report.checks import iter_claims


def _fake_checkers(monkeypatch, fails: dict[str, set[str]], calls: list | None = None) -> None:
    """Replace the model calls: each model fails the locations listed for it in *fails* and passes the rest.

    Every call's (model, locations asked) is appended to *calls* when given.
    """
    def fake_call(model, max_tokens, output_format, **kwargs):
        asked = re.findall(r"=== Claim at (\S+) ", kwargs["messages"][0]["content"])
        if calls is not None:
            calls.append((model, asked))
        verdicts = [Verdict(where=where, supported=where not in fails.get(model, set()), reason=f"{model} says no")
                    for where in asked]
        return StructuredReply(Verdicts(verdicts=verdicts), input_tokens=1000, output_tokens=100)
    monkeypatch.setattr(claim_check, "structured_call", fake_call)


def test_every_cited_claim_is_checked_with_its_passages(monkeypatch):
    calls = []
    _fake_checkers(monkeypatch, {}, calls)
    report = sample_report()
    result = check_claims(report, sample_store())
    assert result.unsupported == {}
    # One Haiku batch covering every claim; no second opinion was needed.
    assert [model for model, _ in calls] == [CHECK_MODEL]
    assert calls[0][1] == [where for where, _ in iter_claims(report)]


def test_a_claim_is_unsupported_only_when_both_models_fail_it(monkeypatch):
    calls = []
    _fake_checkers(monkeypatch, {CHECK_MODEL: {"setup_summary[0]", "key_risks[0]"},
                                 SECOND_MODEL: {"key_risks[0]"}}, calls)
    result = check_claims(sample_report(), sample_store())
    # Sonnet saw only Haiku's failures; it rescued setup_summary[0].
    assert calls[1] == (SECOND_MODEL, ["setup_summary[0]", "key_risks[0]"])
    assert result.unsupported == {"key_risks[0]": f"{SECOND_MODEL} says no"}
    assert result.cost_usd > 0


def test_a_claim_citing_nothing_known_is_left_to_the_id_checks(monkeypatch):
    calls = []
    _fake_checkers(monkeypatch, {}, calls)
    report = sample_report(setup_summary=[claim("Made up.", "Q.999")])
    check_claims(report, sample_store())
    assert "setup_summary[0]" not in calls[0][1]


def test_only_limits_the_check_to_the_named_claims(monkeypatch):
    calls = []
    _fake_checkers(monkeypatch, {}, calls)
    check_claims(sample_report(), sample_store(), only={"guidance[0]"})
    assert calls == [(CHECK_MODEL, ["guidance[0]"])]


def test_a_failed_call_leaves_its_claims_kept_and_noted_as_not_checked(monkeypatch):
    def failing_call(model, max_tokens, output_format, **kwargs):
        raise NoStructuredOutput(model, "max_tokens", input_tokens=5000, output_tokens=32000)
    monkeypatch.setattr(claim_check, "structured_call", failing_call)
    report = sample_report()
    result = check_claims(report, sample_store())
    assert result.unsupported == {}
    assert result.cost_usd > 0   # the cut-off call was still paid for
    assert len(report.check_notes) == len(list(iter_claims(report)))
    assert report.check_notes[0].startswith("setup_summary[0]: not checked against its passages — "
                                            f"{CHECK_MODEL} call failed (NoStructuredOutput")


def test_a_claim_the_model_skips_is_noted_never_passed(monkeypatch):
    def skipping_call(model, max_tokens, output_format, **kwargs):
        return StructuredReply(Verdicts(verdicts=[Verdict(where="guidance[0]", supported=True, reason="ok"),
                                                  Verdict(where="invented[9]", supported=False, reason="?")]),
                               input_tokens=10, output_tokens=10)
    monkeypatch.setattr(claim_check, "structured_call", skipping_call)
    report = sample_report()
    check_claims(report, sample_store())
    assert f"setup_summary[0]: not checked against its passages — {CHECK_MODEL} returned no verdict for it." \
        in report.check_notes
    assert not any(note.startswith("guidance[0]") for note in report.check_notes)


def test_batches_stay_under_the_haiku_price_line():
    long_passage_items = [
        claim_check.ClaimItem(f"setup_summary[{n}]", claim("x", "Q.001"),
                          [sample_store().passages[0].model_copy(update={"text": "y" * 60_000})])
        for n in range(12)
    ]
    batches = claim_check._batches(long_passage_items)
    assert len(batches) > 1
    assert sum(len(batch) for batch in batches) == 12
    for batch in batches:
        chars = sum(len(claim_check.render_claim(item)) for item in batch) + len(claim_check._INSTRUCTIONS)
        assert chars / claim_check._CHARS_PER_TOKEN < claim_check._HAIKU_PRICE_LINE


def test_a_rewrite_lands_only_at_an_unsupported_claim():
    report = sample_report()
    unsupported = {"setup_summary[0]": "the 9% is for a different quarter"}
    repairs = Repairs(repairs=[
        Repair(where="setup_summary[0]", action="rewrite", claim=claim("Revenue was $11.5B.", "F.01")),
        Repair(where="key_risks[0]", action="rewrite", claim=claim("A new claim slipped in.", "K.001")),
    ])
    assert apply_repairs(report, repairs, unsupported) == ["setup_summary[0]"]
    assert report.setup_summary[0].text == "Revenue was $11.5B."
    assert report.setup_summary[0].evidence_ids == ["F.01"]
    assert report.key_risks[0].claim.text == "Competition for members' time."   # untouched
    assert "key_risks[0]: repair ignored — only unsupported claims may be repaired." in report.check_notes
    assert len(list(iter_claims(report))) == len(list(iter_claims(sample_report())))   # nothing added


def test_a_delete_or_an_empty_rewrite_changes_nothing_until_removal():
    report = sample_report()
    unsupported = {"setup_summary[0]": "a", "guidance[0]": "b"}
    repairs = Repairs(repairs=[Repair(where="setup_summary[0]", action="delete", claim=None),
                               Repair(where="guidance[0]", action="rewrite", claim=None)])
    assert apply_repairs(report, repairs, unsupported) == []
    assert report.model_dump() == sample_report().model_dump()


def test_removal_drops_list_claims_empties_optional_slots_and_removes_items():
    report = sample_report(setup_summary=[claim("First.", "Q.001"), claim("Second.", "Q.001")])
    remove_claims(report, {
        "setup_summary[0]": "wrong period",
        "metrics_to_model[0].consensus": "not in the row",
        "key_debates[0].bull": "no such guidance",
        "key_debates[0].bear": "overstated",
    })
    assert [c.text for c in report.setup_summary] == ["Second."]
    assert report.metrics_to_model[0].consensus is None
    assert report.metrics_to_model[0].view is not None
    assert report.key_debates == []   # one removal of the debate, though both sides failed
    assert "key_debates[0].bull: its whole key_debates[0] item was removed — its sources don't support it " \
           "(no such guidance)." in report.check_notes
    assert "A claim was removed because its sources don't support it: 'First.' (wrong period)." in report.data_gaps
    assert len(report.check_notes) == 4


def test_removing_two_items_of_one_section_removes_the_right_ones():
    report = sample_report(setup_summary=[claim(f"Claim {n}.", "Q.001") for n in range(4)])
    remove_claims(report, {"setup_summary[1]": "x", "setup_summary[3]": "y"})
    assert [c.text for c in report.setup_summary] == ["Claim 0.", "Claim 2."]


def test_removing_an_unknown_location_fails_loudly():
    with pytest.raises(ValueError, match="not a claim"):
        remove_claims(sample_report(), {"key_debates[5].bull": "x"})
