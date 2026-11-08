"""Generate a synthetic but realistic insurance policy corpus with exact ground truth.

Why synthetic: a RAG evaluation harness is only as good as its labels. Real policy
wordings would force hand-labelling of which clause answers which question, which is
slow, subjective, and impossible to keep in sync as the corpus grows. Here every
clause is authored alongside the questions it answers, so retrieval ground truth is
exact by construction and the eval set regenerates deterministically.

The wordings imitate the structure and register of Indian general/life insurance
policy documents (clause numbering, defined terms, exclusion schedules, waiting
periods). Insurers, products and UINs are invented.

Outputs:
    data/corpus/<policy_id>.txt      rendered policy wording
    data/corpus/manifest.json        policy metadata + flattened clause index
    data/eval/golden_qa.json         question set with gold clause ids

Usage:
    python scripts/generate_corpus.py --out-dir data
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass
class QA:
    """A question this clause answers, with its gold answer."""

    question: str
    answer: str
    kind: str  # factual | boolean | procedural | numeric
    aliases: list[str] = field(default_factory=list)
    # Acceptable answer strings for the deterministic scorer, beyond `answer`.
    answer_keys: list[str] = field(default_factory=list)


@dataclass
class Clause:
    number: str  # "4.2"
    title: str
    text: str
    qa: list[QA] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    number: str  # "4"
    title: str
    clauses: list[Clause]


@dataclass
class Policy:
    policy_id: str
    insurer: str
    product_name: str
    product_type: str
    uin: str
    version: str
    sections: list[Section]

    @property
    def display(self) -> str:
        # Citations read better with the product name alone; the insurer travels
        # separately in chunk metadata and is still available to the reranker.
        return self.product_name


def _clause_id(policy_id: str, clause_number: str) -> str:
    return f"{policy_id}#{clause_number}"


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_policy(p: Policy) -> str:
    lines: list[str] = []
    lines.append(p.insurer.upper())
    lines.append(p.product_name.upper())
    lines.append(f"Policy Wording | UIN: {p.uin} | Version {p.version}")
    lines.append(f"Product Type: {p.product_type}")
    lines.append("=" * 88)
    lines.append("")
    for section in p.sections:
        lines.append(f"SECTION {section.number}. {section.title.upper()}")
        lines.append("")
        for clause in section.clauses:
            lines.append(f"{clause.number} {clause.title}")
            body = re.sub(r"\s+", " ", clause.text).strip()
            lines.append(_wrap(body, width=96))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _wrap(text: str, width: int = 96) -> str:
    words = text.split()
    out: list[str] = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Policy 1 - Health indemnity
# --------------------------------------------------------------------------------------


def build_health_aarogya() -> Policy:
    pid = "HLT-AAROGYA"
    sections = [
        Section(
            "1",
            "Preamble and Operative Clause",
            [
                Clause(
                    "1.1",
                    "Contract of Insurance",
                    "This Policy is a contract of insurance issued by Meridian General Insurance Company "
                    "Limited to the Policyholder named in the Policy Schedule. The proposal form, the "
                    "declarations made therein and the Policy Schedule together form the basis of this "
                    "contract. The Company agrees, subject to the terms, conditions, exclusions and "
                    "limitations contained herein, to indemnify the Insured Person for Medical Expenses "
                    "reasonably and necessarily incurred during the Policy Period, up to the Sum Insured.",
                ),
                Clause(
                    "1.2",
                    "Territorial Scope and Governing Law",
                    "Cover under this Policy operates within the geographical limits of India. All claims "
                    "shall be payable in Indian Rupees. This Policy is governed by and shall be construed "
                    "in accordance with the laws of India, and the courts at the place of issuance shall "
                    "have exclusive jurisdiction over any dispute arising out of this Policy.",
                    qa=[
                        QA(
                            "Does the Aarogya Suraksha health policy cover treatment taken outside India?",
                            "No. Cover under the Aarogya Suraksha policy operates only within the geographical limits of India.",
                            "boolean",
                            answer_keys=["no", "only within india", "india"],
                        )
                    ],
                ),
            ],
        ),
        Section(
            "2",
            "Definitions",
            [
                Clause(
                    "2.1",
                    "Hospitalisation",
                    "Hospitalisation means admission of an Insured Person into a Hospital for a continuous "
                    "period of at least twenty-four (24) consecutive hours for Medically Necessary Treatment, "
                    "save that this minimum duration shall not apply to Day Care Treatment listed in Annexure "
                    "A of this Policy.",
                    qa=[
                        QA(
                            "What is the minimum hospitalisation duration under the Aarogya Suraksha policy?",
                            "24 consecutive hours, except for listed Day Care Treatments where the minimum duration does not apply.",
                            "numeric",
                            answer_keys=["24", "twenty-four", "24 hours"],
                        )
                    ],
                    entities={"definition": "Hospitalisation", "min_hours": 24},
                ),
                Clause(
                    "2.2",
                    "Pre-Existing Disease",
                    "Pre-Existing Disease means any condition, ailment, injury or disease that is diagnosed "
                    "by a Medical Practitioner within forty-eight (48) months prior to the effective date of "
                    "the first Policy issued by the Company to the Insured Person, or for which medical advice "
                    "or treatment was recommended by or received from a Medical Practitioner within forty-eight "
                    "(48) months prior to that date.",
                    qa=[
                        QA(
                            "How does the Aarogya Suraksha policy define a pre-existing disease?",
                            "A condition diagnosed, or for which medical advice or treatment was recommended or received, within 48 months before the effective date of the first policy issued to the Insured Person.",
                            "factual",
                            answer_keys=["48 months", "forty-eight"],
                        )
                    ],
                    entities={"definition": "Pre-Existing Disease", "lookback_months": 48},
                ),
                Clause(
                    "2.3",
                    "Medically Necessary Treatment",
                    "Medically Necessary Treatment means any treatment, test, medication or stay in Hospital "
                    "which is required for the medical management of the Illness or Injury suffered by the "
                    "Insured Person, which is appropriate to the standard medical practice prevailing in India, "
                    "which must not exceed the level of care necessary to provide safe and adequate treatment, "
                    "and which is prescribed by a qualified Medical Practitioner.",
                ),
                Clause(
                    "2.4",
                    "Network Provider",
                    "Network Provider means a Hospital enlisted by the Company or by its Third Party "
                    "Administrator to provide medical services to Insured Persons on a cashless basis. The "
                    "current list of Network Providers is published on the Company's website and is subject "
                    "to revision without prior notice.",
                ),
            ],
        ),
        Section(
            "3",
            "Scope of Cover",
            [
                Clause(
                    "3.1",
                    "In-Patient Hospitalisation Expenses",
                    "The Company shall indemnify Medical Expenses incurred on Hospitalisation of the Insured "
                    "Person during the Policy Period, including room, boarding and nursing charges, Intensive "
                    "Care Unit charges, surgeon, anaesthetist and specialist fees, anaesthesia, blood, oxygen, "
                    "operation theatre charges, surgical appliances, medicines and drugs, and diagnostic "
                    "procedures, up to the Sum Insured specified in the Policy Schedule.",
                ),
                Clause(
                    "3.2",
                    "Pre-Hospitalisation and Post-Hospitalisation Expenses",
                    "Medical Expenses incurred during the sixty (60) days immediately preceding the date of "
                    "Hospitalisation and the ninety (90) days immediately following the date of discharge "
                    "shall be payable, provided the Hospitalisation claim itself has been admitted by the "
                    "Company and the expenses relate to the same condition.",
                    qa=[
                        QA(
                            "What is the pre-hospitalisation and post-hospitalisation cover period under the Aarogya Suraksha policy?",
                            "60 days of pre-hospitalisation expenses and 90 days of post-hospitalisation expenses.",
                            "numeric",
                            answer_keys=["60", "90", "sixty", "ninety"],
                        )
                    ],
                    entities={"pre_hosp_days": 60, "post_hosp_days": 90},
                ),
                Clause(
                    "3.3",
                    "Day Care Treatment",
                    "The Company shall indemnify expenses incurred on the Day Care Treatments listed in "
                    "Annexure A which, owing to technological advancement, require less than twenty-four (24) "
                    "hours of Hospitalisation. Treatment taken in the out-patient department of a Hospital "
                    "shall not qualify as Day Care Treatment under this clause.",
                ),
                Clause(
                    "3.4",
                    "Road Ambulance Charges",
                    "Expenses incurred on transportation of the Insured Person by road ambulance to a Hospital "
                    "following an Emergency, or between Hospitals for the purpose of obtaining Medically "
                    "Necessary Treatment, shall be reimbursed up to Rs. 5,000 per Hospitalisation event.",
                    qa=[
                        QA(
                            "What is the road ambulance limit under the Aarogya Suraksha policy?",
                            "Rs. 5,000 per hospitalisation event.",
                            "numeric",
                            answer_keys=["5,000", "5000", "rs. 5,000"],
                        )
                    ],
                    entities={"benefit": "Road Ambulance", "limit_inr": 5000},
                ),
                Clause(
                    "3.5",
                    "AYUSH Treatment",
                    "Medical Expenses incurred on in-patient treatment taken under Ayurveda, Yoga, Naturopathy, "
                    "Unani, Siddha and Homeopathy systems of medicine in a Hospital recognised by the relevant "
                    "statutory authority shall be indemnified up to twenty-five percent (25%) of the Sum "
                    "Insured per Policy Year.",
                    qa=[
                        QA(
                            "Is AYUSH treatment covered under the Aarogya Suraksha policy and up to what limit?",
                            "Yes. In-patient AYUSH treatment at a recognised hospital is covered up to 25% of the Sum Insured per policy year.",
                            "boolean",
                            answer_keys=["25%", "25 percent", "yes"],
                        )
                    ],
                    entities={"benefit": "AYUSH", "limit_pct_si": 25},
                ),
                Clause(
                    "3.6",
                    "Organ Donor Expenses",
                    "The Company shall indemnify Medical Expenses incurred towards the organ donor's "
                    "Hospitalisation for harvesting of the organ, provided the organ is for the use of the "
                    "Insured Person and the claim for the Insured Person's transplant has been admitted. "
                    "Pre-harvesting screening expenses of the donor and the cost of the organ itself are "
                    "not payable.",
                ),
            ],
        ),
        Section(
            "4",
            "Waiting Periods",
            [
                Clause(
                    "4.1",
                    "Initial Waiting Period",
                    "Save in respect of claims arising out of an Accident, the Company shall not be liable "
                    "for any expenses incurred in connection with Illness contracted during the first thirty "
                    "(30) days from the Policy Inception Date. This waiting period shall not apply to "
                    "subsequent renewals of the Policy without a break.",
                    qa=[
                        QA(
                            "What is the initial waiting period under the Aarogya Suraksha policy?",
                            "30 days from policy inception, except for accidental injury.",
                            "numeric",
                            answer_keys=["30 days", "thirty"],
                        )
                    ],
                    entities={"waiting_period": "Initial", "days": 30},
                ),
                Clause(
                    "4.2",
                    "Pre-Existing Disease Waiting Period",
                    "Expenses related to the treatment of a Pre-Existing Disease and its direct complications "
                    "shall be excluded until the expiry of thirty-six (36) months of continuous coverage after "
                    "the date of inception of the first Policy with the Company. Where the Insured Person has "
                    "enhanced the Sum Insured after inception, this waiting period shall apply afresh to the "
                    "enhanced portion of the Sum Insured.",
                    qa=[
                        QA(
                            "What is the pre-existing disease waiting period under the Aarogya Suraksha policy?",
                            "36 months of continuous coverage from the inception of the first policy with the Company.",
                            "numeric",
                            answer_keys=["36 months", "thirty-six"],
                        ),
                        QA(
                            "If I increase my sum insured on the Aarogya Suraksha policy, does the pre-existing disease waiting period restart?",
                            "Yes. The 36-month pre-existing disease waiting period applies afresh to the enhanced portion of the Sum Insured.",
                            "boolean",
                            answer_keys=["yes", "afresh", "enhanced portion"],
                        ),
                    ],
                    entities={"waiting_period": "Pre-Existing Disease", "months": 36},
                ),
                Clause(
                    "4.3",
                    "Specified Disease and Procedure Waiting Period",
                    "A waiting period of twenty-four (24) months of continuous coverage shall apply to "
                    "cataract, benign prostatic hypertrophy, hernia of all types, hydrocele, fissures and "
                    "fistula in ano, haemorrhoids, sinusitis, gall bladder and bile duct stone disease, "
                    "varicose veins, and joint replacement surgery other than that necessitated by an Accident.",
                    qa=[
                        QA(
                            "Is cataract surgery covered immediately under the Aarogya Suraksha policy?",
                            "No. Cataract falls under the specified disease waiting period of 24 months of continuous coverage.",
                            "boolean",
                            answer_keys=["24 months", "no", "twenty-four"],
                        )
                    ],
                    entities={"waiting_period": "Specified Disease", "months": 24},
                ),
                Clause(
                    "4.4",
                    "Maternity Waiting Period",
                    "Maternity Expenses, including childbirth and lawful medical termination of pregnancy, "
                    "shall be payable only after the Insured Person has been continuously covered under this "
                    "Policy for a period of thirty-six (36) months. Maternity benefit is limited to two "
                    "deliveries or terminations during the lifetime of the Insured Person.",
                    qa=[
                        QA(
                            "What is the maternity waiting period under the Aarogya Suraksha policy?",
                            "36 months of continuous coverage, and the benefit is limited to two deliveries or terminations in the Insured Person's lifetime.",
                            "numeric",
                            answer_keys=["36 months", "thirty-six", "two deliveries"],
                        )
                    ],
                    entities={"waiting_period": "Maternity", "months": 36, "max_events": 2},
                ),
            ],
        ),
        Section(
            "5",
            "Sub-Limits, Deductible and Co-Payment",
            [
                Clause(
                    "5.1",
                    "Room Rent Sub-Limit",
                    "Room, boarding and nursing charges shall be payable up to one percent (1%) of the Sum "
                    "Insured per day for a normal room, and up to two percent (2%) of the Sum Insured per day "
                    "for an Intensive Care Unit. Where the Insured Person opts for a room category exceeding "
                    "this entitlement, all associated Medical Expenses shall be payable in the same proportion "
                    "that the entitled room rent bears to the actual room rent incurred.",
                    qa=[
                        QA(
                            "What is the room rent sub-limit under the Aarogya Suraksha policy?",
                            "1% of the Sum Insured per day for a normal room and 2% per day for ICU, with proportionate deduction if a higher room category is chosen.",
                            "numeric",
                            answer_keys=["1%", "2%", "one percent", "two percent"],
                        ),
                        QA(
                            "What happens if I choose a hospital room above my entitlement under the Aarogya Suraksha policy?",
                            "All associated medical expenses are reduced proportionately, in the ratio that the entitled room rent bears to the actual room rent incurred.",
                            "procedural",
                            answer_keys=["proportion", "proportionate"],
                        ),
                    ],
                    entities={"sub_limit": "Room Rent", "normal_pct": 1, "icu_pct": 2},
                ),
                Clause(
                    "5.2",
                    "Co-Payment",
                    "A co-payment of ten percent (10%) shall apply to each and every admissible claim where "
                    "the Insured Person was aged fifty-six (56) years or above on the date of first inception "
                    "of the Policy. No co-payment shall apply to Insured Persons who were below that age at "
                    "first inception.",
                    qa=[
                        QA(
                            "Does the Aarogya Suraksha policy have a co-payment?",
                            "Yes, a 10% co-payment applies to every admissible claim where the Insured Person was aged 56 or above at first inception of the policy.",
                            "boolean",
                            answer_keys=["10%", "ten percent", "56"],
                        )
                    ],
                    entities={"co_payment_pct": 10, "age_trigger": 56},
                ),
                Clause(
                    "5.3",
                    "Cataract Sub-Limit",
                    "Expenses on treatment of cataract, once the applicable waiting period has expired, shall "
                    "be payable up to Rs. 40,000 per eye per Policy Year, inclusive of the cost of the "
                    "intraocular lens.",
                    qa=[
                        QA(
                            "What is the cataract sub-limit under the Aarogya Suraksha policy?",
                            "Rs. 40,000 per eye per policy year, inclusive of the intraocular lens.",
                            "numeric",
                            answer_keys=["40,000", "40000"],
                        )
                    ],
                    entities={"sub_limit": "Cataract", "limit_inr": 40000},
                ),
                Clause(
                    "5.4",
                    "Modern Treatment Methods",
                    "Expenses incurred on robotic surgery, stem cell therapy, oral chemotherapy, intra-vitreal "
                    "injections, bronchial thermoplasty, deep brain stimulation and immunotherapy shall be "
                    "payable up to fifty percent (50%) of the Sum Insured per Policy Year, subject to the "
                    "treatment being taken as in-patient or as listed Day Care Treatment.",
                    entities={"sub_limit": "Modern Treatment", "limit_pct_si": 50},
                ),
            ],
        ),
        Section(
            "6",
            "Exclusions",
            [
                Clause(
                    "6.1",
                    "Cosmetic and Plastic Surgery",
                    "Expenses for cosmetic or plastic surgery or any treatment to change appearance are "
                    "excluded, unless such treatment is certified by the attending Medical Practitioner to be "
                    "Medically Necessary for reconstruction following an Accident, Burn or Cancer.",
                    qa=[
                        QA(
                            "Is cosmetic surgery covered under the Aarogya Suraksha policy?",
                            "No, cosmetic or plastic surgery is excluded unless medically necessary for reconstruction following an accident, burn or cancer.",
                            "boolean",
                            answer_keys=["no", "excluded", "unless"],
                        )
                    ],
                    entities={"exclusion": "Cosmetic Surgery"},
                ),
                Clause(
                    "6.2",
                    "Hazardous Activities",
                    "Expenses related to any treatment necessitated due to participation as a professional in "
                    "hazardous or adventure sports, including para-jumping, rock climbing, mountaineering, "
                    "rafting, motor racing, horse racing, scuba diving, hand gliding, sky diving and deep-sea "
                    "diving, are excluded.",
                    qa=[
                        QA(
                            "Is scuba diving injury covered under the Aarogya Suraksha policy?",
                            "No. Treatment necessitated by professional participation in hazardous or adventure sports, including scuba diving, is excluded.",
                            "boolean",
                            answer_keys=["no", "excluded", "hazardous"],
                        )
                    ],
                    entities={"exclusion": "Hazardous Activities"},
                ),
                Clause(
                    "6.3",
                    "Substance Abuse and Self-Inflicted Injury",
                    "Expenses arising from or attributable to the abuse of alcohol or drugs, or treatment "
                    "arising from any intentionally self-inflicted injury, are excluded from this Policy.",
                    entities={"exclusion": "Substance Abuse"},
                ),
                Clause(
                    "6.4",
                    "Non-Medical Consumables",
                    "Items of personal comfort and convenience, including but not limited to television "
                    "charges, telephone charges, internet charges, attendant charges, toiletries, and "
                    "administrative or registration fees, are not payable. A complete schedule of "
                    "non-payable items is set out in Annexure B.",
                    qa=[
                        QA(
                            "Are attendant charges payable under the Aarogya Suraksha policy?",
                            "No. Attendant charges are listed among non-medical items of personal comfort and convenience which are not payable.",
                            "boolean",
                            answer_keys=["no", "not payable", "excluded"],
                        )
                    ],
                    entities={"exclusion": "Non-Medical Consumables"},
                ),
                Clause(
                    "6.5",
                    "Out-Patient Treatment",
                    "Expenses on treatment taken in the out-patient department of a Hospital or a clinic, "
                    "including consultations, diagnostics and pharmacy purchases not connected with an "
                    "admissible Hospitalisation, are excluded unless an out-patient benefit has been "
                    "specifically opted for and is shown in the Policy Schedule.",
                    entities={"exclusion": "Out-Patient Treatment"},
                ),
                Clause(
                    "6.6",
                    "War and Nuclear Perils",
                    "The Company shall not be liable for any claim arising directly or indirectly out of war, "
                    "invasion, act of foreign enemy, hostilities, civil war, rebellion, revolution, military "
                    "or usurped power, or from nuclear weapons material or ionising radiation.",
                    entities={"exclusion": "War and Nuclear Perils"},
                ),
            ],
        ),
        Section(
            "7",
            "Claims Procedure",
            [
                Clause(
                    "7.1",
                    "Notification of Claim",
                    "In the event of planned Hospitalisation, the Company or its Third Party Administrator "
                    "shall be notified at least forty-eight (48) hours prior to admission. In the event of "
                    "emergency Hospitalisation, notification shall be given within twenty-four (24) hours of "
                    "admission.",
                    qa=[
                        QA(
                            "How soon must I notify a planned hospitalisation under the Aarogya Suraksha policy?",
                            "At least 48 hours before admission for planned hospitalisation, and within 24 hours of admission in an emergency.",
                            "numeric",
                            answer_keys=["48 hours", "24 hours", "forty-eight"],
                        )
                    ],
                    entities={"notify_planned_hours": 48, "notify_emergency_hours": 24},
                ),
                Clause(
                    "7.2",
                    "Submission of Documents",
                    "All documents in support of a reimbursement claim, including the original discharge "
                    "summary, hospital bills, payment receipts, investigation reports and the duly completed "
                    "claim form, shall be submitted to the Company within thirty (30) days of the date of "
                    "discharge from the Hospital.",
                    qa=[
                        QA(
                            "Within how many days must claim documents be submitted under the Aarogya Suraksha policy?",
                            "Within 30 days of the date of discharge from hospital.",
                            "numeric",
                            answer_keys=["30 days", "thirty"],
                        )
                    ],
                    entities={"doc_submission_days": 30},
                ),
                Clause(
                    "7.3",
                    "Cashless Facility",
                    "Cashless facility is available only at Network Providers. A pre-authorisation request "
                    "shall be submitted by the Hospital to the Third Party Administrator, who shall respond "
                    "with an approval, query or denial within four (4) hours of receiving a complete request. "
                    "Denial of pre-authorisation does not constitute denial of the claim, and the Insured "
                    "Person may file for reimbursement.",
                    qa=[
                        QA(
                            "How long does pre-authorisation take under the Aarogya Suraksha policy?",
                            "The third party administrator responds within 4 hours of receiving a complete pre-authorisation request.",
                            "numeric",
                            answer_keys=["4 hours", "four hours"],
                        )
                    ],
                    entities={"preauth_response_hours": 4},
                ),
                Clause(
                    "7.4",
                    "Settlement and Interest",
                    "The Company shall settle or reject a claim within thirty (30) days of receipt of the "
                    "last necessary document. Where settlement is delayed beyond this period, the Company "
                    "shall pay interest at a rate two percent (2%) above the prevailing bank rate from the "
                    "date of receipt of the last necessary document until the date of payment.",
                    qa=[
                        QA(
                            "What interest is payable on delayed claim settlement under the Aarogya Suraksha policy?",
                            "Interest at 2% above the prevailing bank rate, from receipt of the last necessary document until payment, where settlement exceeds 30 days.",
                            "numeric",
                            answer_keys=["2%", "two percent", "bank rate"],
                        )
                    ],
                    entities={"settlement_days": 30, "delay_interest_pct_over_bank_rate": 2},
                ),
            ],
        ),
        Section(
            "8",
            "General Conditions",
            [
                Clause(
                    "8.1",
                    "Free Look Period",
                    "The Insured Person may cancel this Policy within fifteen (15) days of receipt of the "
                    "Policy document, stating the reasons for cancellation, and shall be refunded the premium "
                    "paid less any expenses incurred by the Company on medical examination and stamp duty. "
                    "The free look period shall be thirty (30) days where the Policy has been purchased "
                    "through distance marketing.",
                    qa=[
                        QA(
                            "What is the free look period under the Aarogya Suraksha policy?",
                            "15 days from receipt of the policy document, extended to 30 days for policies purchased through distance marketing.",
                            "numeric",
                            answer_keys=["15 days", "fifteen", "30 days"],
                        )
                    ],
                    entities={"free_look_days": 15, "free_look_days_distance": 30},
                ),
                Clause(
                    "8.2",
                    "Grace Period",
                    "A grace period of thirty (30) days from the Policy expiry date is allowed for payment of "
                    "renewal premium without loss of continuity benefits such as waiting periods and coverage "
                    "of Pre-Existing Diseases. Coverage shall not be available for the period during which no "
                    "premium has been received.",
                    qa=[
                        QA(
                            "What is the grace period for renewal under the Aarogya Suraksha policy?",
                            "30 days from the policy expiry date, during which continuity benefits are preserved but no cover applies.",
                            "numeric",
                            answer_keys=["30 days", "thirty"],
                        )
                    ],
                    entities={"grace_period_days": 30},
                ),
                Clause(
                    "8.3",
                    "Cumulative Bonus",
                    "For every claim-free Policy Year, the Sum Insured shall be increased by ten percent "
                    "(10%) of the base Sum Insured, subject to a maximum accumulation of fifty percent (50%). "
                    "In the event of a claim, the accrued cumulative bonus shall be reduced by ten percent "
                    "(10%) of the base Sum Insured at the next renewal.",
                    qa=[
                        QA(
                            "How does the cumulative bonus work under the Aarogya Suraksha policy?",
                            "The Sum Insured increases by 10% of the base Sum Insured for each claim-free year, capped at 50%, and reduces by 10% at the next renewal following a claim.",
                            "numeric",
                            answer_keys=["10%", "50%", "ten percent"],
                        )
                    ],
                    entities={"cb_accrual_pct": 10, "cb_max_pct": 50},
                ),
                Clause(
                    "8.4",
                    "Portability",
                    "The Insured Person may port this Policy to another insurer at renewal by applying to the "
                    "new insurer at least forty-five (45) days but not earlier than sixty (60) days before "
                    "the renewal date. Credit gained for waiting periods shall be transferred to the extent "
                    "of the Sum Insured under the expiring policy.",
                    qa=[
                        QA(
                            "How far in advance must I apply to port the Aarogya Suraksha policy?",
                            "At least 45 days, and not earlier than 60 days, before the renewal date.",
                            "numeric",
                            answer_keys=["45 days", "60 days", "forty-five"],
                        )
                    ],
                    entities={"portability_min_days": 45, "portability_max_days": 60},
                ),
                Clause(
                    "8.5",
                    "Moratorium Period",
                    "After sixty (60) continuous months of coverage have elapsed since inception of the first "
                    "Policy, no claim shall be contestable by the Company on grounds of non-disclosure or "
                    "misrepresentation, except on grounds of established fraud. This moratorium shall apply "
                    "to the Sum Insured of the first Policy, and separately to any enhanced Sum Insured from "
                    "the date of such enhancement.",
                    qa=[
                        QA(
                            "What is the moratorium period under the Aarogya Suraksha policy?",
                            "60 continuous months, after which claims cannot be contested for non-disclosure or misrepresentation except on grounds of established fraud.",
                            "numeric",
                            answer_keys=["60 months", "sixty"],
                        )
                    ],
                    entities={"moratorium_months": 60},
                ),
                Clause(
                    "8.6",
                    "Grievance Redressal",
                    "Any grievance may be addressed to the Grievance Officer of the Company, who shall respond "
                    "within fifteen (15) days of receipt. If the response is unsatisfactory, or if no response "
                    "is received, the Insured Person may approach the Insurance Ombudsman having jurisdiction.",
                    entities={"grievance_response_days": 15},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Meridian General Insurance Company Limited",
        product_name="Aarogya Suraksha Health Insurance",
        product_type="Health Indemnity",
        uin="MGICHLIP24001V012425",
        version="2.1",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Policy 2 - Health indemnity (competitor, deliberately similar -> hard negatives)
# --------------------------------------------------------------------------------------


def build_health_nirogam() -> Policy:
    pid = "HLT-NIROGAM"
    sections = [
        Section(
            "1",
            "Preamble",
            [
                Clause(
                    "1.1",
                    "Operative Clause",
                    "Sentinel Health Assurance Limited agrees to indemnify the Insured Person against Medical "
                    "Expenses incurred during the Policy Period in respect of Illness or Injury, subject to "
                    "the Sum Insured, the terms set out herein and the particulars stated in the Policy "
                    "Schedule, in consideration of the premium received.",
                )
            ],
        ),
        Section(
            "2",
            "Definitions",
            [
                Clause(
                    "2.1",
                    "Pre-Existing Disease",
                    "Pre-Existing Disease means any condition, ailment, injury or disease diagnosed by a "
                    "Medical Practitioner within thirty-six (36) months prior to the effective date of the "
                    "first Policy issued by the Insurer, or for which medical advice or treatment was "
                    "recommended by or received from a Medical Practitioner within thirty-six (36) months "
                    "prior to that date.",
                    entities={"definition": "Pre-Existing Disease", "lookback_months": 36},
                ),
                Clause(
                    "2.2",
                    "Hospital",
                    "Hospital means an institution established for in-patient care and day care treatment "
                    "which is registered with the local authorities, has qualified nursing staff under its "
                    "employment round the clock, has at least ten (10) in-patient beds in towns with a "
                    "population of less than ten lakhs and fifteen (15) in-patient beds elsewhere, has a "
                    "fully equipped operation theatre of its own, and maintains daily records of patients.",
                    qa=[
                        QA(
                            "How many in-patient beds must a hospital have to qualify under the Nirogam Plus policy?",
                            "At least 10 beds in towns with a population below ten lakhs, and 15 beds elsewhere.",
                            "numeric",
                            answer_keys=["10", "15", "ten", "fifteen"],
                        )
                    ],
                    entities={"definition": "Hospital", "beds_small_town": 10, "beds_other": 15},
                ),
            ],
        ),
        Section(
            "3",
            "Benefits",
            [
                Clause(
                    "3.1",
                    "In-Patient Treatment",
                    "The Insurer shall indemnify reasonable and customary Medical Expenses incurred on "
                    "in-patient treatment of the Insured Person, including room and nursing charges, "
                    "consultant fees, operation theatre charges, drugs, implants and diagnostics, up to the "
                    "Sum Insured stated in the Policy Schedule.",
                ),
                Clause(
                    "3.2",
                    "Pre and Post Hospitalisation",
                    "Expenses incurred in the thirty (30) days preceding Hospitalisation and the sixty (60) "
                    "days following discharge shall be indemnified, provided they relate to the condition "
                    "for which the admitted Hospitalisation claim was made.",
                    qa=[
                        QA(
                            "What is the pre and post hospitalisation cover under the Nirogam Plus policy?",
                            "30 days of pre-hospitalisation expenses and 60 days of post-hospitalisation expenses.",
                            "numeric",
                            answer_keys=["30", "60", "thirty", "sixty"],
                        )
                    ],
                    entities={"pre_hosp_days": 30, "post_hosp_days": 60},
                ),
                Clause(
                    "3.3",
                    "Domiciliary Hospitalisation",
                    "Where treatment that would ordinarily require Hospitalisation is taken at home because "
                    "the condition of the Insured Person prevents removal to a Hospital, or because a hospital "
                    "bed was unavailable, such expenses shall be indemnified provided the treatment continues "
                    "for at least three (3) consecutive days.",
                    qa=[
                        QA(
                            "Does the Nirogam Plus policy cover domiciliary hospitalisation?",
                            "Yes, where the patient cannot be moved to hospital or no bed was available, provided treatment continues for at least 3 consecutive days.",
                            "boolean",
                            answer_keys=["yes", "3 days", "three"],
                        )
                    ],
                    entities={"benefit": "Domiciliary Hospitalisation", "min_days": 3},
                ),
                Clause(
                    "3.4",
                    "Annual Health Check-Up",
                    "The Insured Person shall be entitled to a preventive health check-up at a Network "
                    "Provider at the end of every two (2) continuous claim-free Policy Years, up to one "
                    "percent (1%) of the Sum Insured subject to a maximum of Rs. 7,500.",
                    qa=[
                        QA(
                            "Is a health check-up included in the Nirogam Plus policy?",
                            "Yes, after every 2 continuous claim-free policy years, up to 1% of the Sum Insured capped at Rs. 7,500.",
                            "boolean",
                            answer_keys=["yes", "7,500", "7500", "two"],
                        )
                    ],
                    entities={"benefit": "Health Check-Up", "cap_inr": 7500, "every_years": 2},
                ),
                Clause(
                    "3.5",
                    "Road Ambulance",
                    "Ambulance charges for transporting the Insured Person to the nearest Hospital following "
                    "an Emergency shall be reimbursed up to Rs. 3,000 per event.",
                    entities={"benefit": "Road Ambulance", "limit_inr": 3000},
                ),
            ],
        ),
        Section(
            "4",
            "Waiting Periods",
            [
                Clause(
                    "4.1",
                    "Initial Waiting Period",
                    "No benefit shall be payable in respect of Illness contracted within the first thirty "
                    "(30) days of the Policy Inception Date, other than claims arising from an Accident.",
                    entities={"waiting_period": "Initial", "days": 30},
                ),
                Clause(
                    "4.2",
                    "Pre-Existing Disease Waiting Period",
                    "Expenses related to the treatment of a Pre-Existing Disease and its direct complications "
                    "shall be excluded until the expiry of twenty-four (24) months of continuous coverage from "
                    "the inception of the first Policy with the Insurer.",
                    qa=[
                        QA(
                            "What is the pre-existing disease waiting period under the Nirogam Plus policy?",
                            "24 months of continuous coverage from inception of the first policy with the Insurer.",
                            "numeric",
                            answer_keys=["24 months", "twenty-four"],
                        )
                    ],
                    entities={"waiting_period": "Pre-Existing Disease", "months": 24},
                ),
                Clause(
                    "4.3",
                    "Specified Ailments",
                    "A waiting period of eighteen (18) months shall apply to cataract, hernia, piles, sinusitis, "
                    "gall stones, kidney stones, hysterectomy and surgery of the knee or hip other than that "
                    "necessitated by an Accident.",
                    entities={"waiting_period": "Specified Disease", "months": 18},
                ),
                Clause(
                    "4.4",
                    "Maternity",
                    "Maternity Expenses shall be payable after twenty-four (24) months of continuous coverage, "
                    "limited to Rs. 50,000 for a normal delivery and Rs. 75,000 for a caesarean section, and "
                    "restricted to the first two living children.",
                    qa=[
                        QA(
                            "What are the maternity limits under the Nirogam Plus policy?",
                            "Rs. 50,000 for a normal delivery and Rs. 75,000 for a caesarean, after a 24-month waiting period and restricted to the first two living children.",
                            "numeric",
                            answer_keys=["50,000", "75,000", "24 months"],
                        )
                    ],
                    entities={
                        "waiting_period": "Maternity",
                        "months": 24,
                        "normal_inr": 50000,
                        "caesarean_inr": 75000,
                    },
                ),
            ],
        ),
        Section(
            "5",
            "Limits and Co-Payment",
            [
                Clause(
                    "5.1",
                    "Room Rent",
                    "Room and nursing charges shall be restricted to a Single Private Air-Conditioned Room. "
                    "Where a higher category is availed, the Insurer shall be liable only in the proportion "
                    "that the eligible room rent bears to the actual room rent charged.",
                    qa=[
                        QA(
                            "What room category am I entitled to under the Nirogam Plus policy?",
                            "A single private air-conditioned room; a higher category triggers proportionate reduction of the claim.",
                            "factual",
                            answer_keys=["single private", "air-conditioned", "proportion"],
                        )
                    ],
                    entities={"sub_limit": "Room Rent", "category": "Single Private AC Room"},
                ),
                Clause(
                    "5.2",
                    "Zone-Based Co-Payment",
                    "Where the Insured Person obtains treatment in a city classified by the Insurer as Zone A "
                    "but has paid premium applicable to Zone B, a co-payment of twenty percent (20%) shall "
                    "apply to the admissible claim amount. No co-payment applies where treatment is taken "
                    "within the zone of the premium paid.",
                    qa=[
                        QA(
                            "When does the 20% co-payment apply under the Nirogam Plus policy?",
                            "When the Insured Person is treated in a Zone A city but paid Zone B premium.",
                            "procedural",
                            answer_keys=["20%", "zone a", "twenty percent"],
                        )
                    ],
                    entities={"co_payment_pct": 20, "trigger": "zone mismatch"},
                ),
            ],
        ),
        Section(
            "6",
            "Exclusions",
            [
                Clause(
                    "6.1",
                    "Dental Treatment",
                    "Dental treatment or surgery of any kind, unless necessitated by an Accident and requiring "
                    "Hospitalisation, is excluded. Routine examinations, scaling, filling and orthodontic "
                    "treatment are not payable under any circumstances.",
                    qa=[
                        QA(
                            "Is dental treatment covered under the Nirogam Plus policy?",
                            "No, except where necessitated by an accident and requiring hospitalisation. Routine dental work and orthodontics are never payable.",
                            "boolean",
                            answer_keys=["no", "excluded", "accident"],
                        )
                    ],
                    entities={"exclusion": "Dental Treatment"},
                ),
                Clause(
                    "6.2",
                    "Obesity and Weight Control",
                    "Treatment for obesity or weight control is excluded, save that bariatric surgery shall be "
                    "payable where the Body Mass Index of the Insured Person is greater than forty (40), or "
                    "greater than thirty-five (35) with an associated co-morbidity, and the surgery is advised "
                    "in writing by a Medical Practitioner.",
                    qa=[
                        QA(
                            "Is bariatric surgery covered under the Nirogam Plus policy?",
                            "Yes, where BMI exceeds 40, or exceeds 35 with an associated co-morbidity, and the surgery is advised in writing by a medical practitioner.",
                            "boolean",
                            answer_keys=["40", "35", "yes", "bmi"],
                        )
                    ],
                    entities={"exclusion": "Obesity", "bmi_threshold": 40, "bmi_comorbid": 35},
                ),
                Clause(
                    "6.3",
                    "Unproven Treatments",
                    "Expenses related to any treatment, including drugs and devices, which is not supported "
                    "by established medical practice and documented evidence of effectiveness, are excluded "
                    "from this Policy.",
                    entities={"exclusion": "Unproven Treatments"},
                ),
                Clause(
                    "6.4",
                    "Breach of Law",
                    "Treatment for any Illness or Injury resulting from the Insured Person committing or "
                    "attempting to commit a breach of law with criminal intent is excluded.",
                    entities={"exclusion": "Breach of Law"},
                ),
            ],
        ),
        Section(
            "7",
            "Claims and Conditions",
            [
                Clause(
                    "7.1",
                    "Claim Intimation",
                    "Intimation of a claim shall be given to the Insurer within twenty-four (24) hours of "
                    "emergency admission and at least seventy-two (72) hours before a planned admission.",
                    qa=[
                        QA(
                            "How far in advance must a planned admission be intimated under the Nirogam Plus policy?",
                            "At least 72 hours before planned admission, and within 24 hours of an emergency admission.",
                            "numeric",
                            answer_keys=["72 hours", "seventy-two", "24 hours"],
                        )
                    ],
                    entities={"notify_planned_hours": 72, "notify_emergency_hours": 24},
                ),
                Clause(
                    "7.2",
                    "Document Submission",
                    "Claim documents shall be submitted within fifteen (15) days of discharge. The Insurer may "
                    "condone a delay where the Insured Person proves to the satisfaction of the Insurer that "
                    "the delay was for reasons beyond their control.",
                    qa=[
                        QA(
                            "What is the claim document submission deadline under the Nirogam Plus policy?",
                            "15 days from discharge, with delay condonable if shown to be beyond the insured's control.",
                            "numeric",
                            answer_keys=["15 days", "fifteen"],
                        )
                    ],
                    entities={"doc_submission_days": 15},
                ),
                Clause(
                    "7.3",
                    "Free Look Period",
                    "This Policy may be cancelled within fifteen (15) days of receipt of the Policy document. "
                    "Where the Policy has been sold through distance marketing, the free look period shall be "
                    "thirty (30) days.",
                    entities={"free_look_days": 15, "free_look_days_distance": 30},
                ),
                Clause(
                    "7.4",
                    "Grace Period",
                    "A grace period of fifteen (15) days is allowed for renewal premium payment where premium "
                    "is paid annually, and thirty (30) days where premium is paid in instalments. Continuity "
                    "of waiting period credit is preserved if renewal occurs within the grace period.",
                    qa=[
                        QA(
                            "What is the grace period under the Nirogam Plus policy?",
                            "15 days for annual premium payment and 30 days where premium is paid in instalments.",
                            "numeric",
                            answer_keys=["15 days", "30 days", "fifteen"],
                        )
                    ],
                    entities={"grace_period_days": 15, "grace_period_days_instalment": 30},
                ),
                Clause(
                    "7.5",
                    "No Claim Discount",
                    "A discount of five percent (5%) on the renewal premium shall be allowed for each "
                    "claim-free Policy Year, subject to a maximum discount of twenty-five percent (25%). "
                    "This discount is forfeited entirely in the event of a claim.",
                    qa=[
                        QA(
                            "What no claim discount does the Nirogam Plus policy give?",
                            "5% off renewal premium per claim-free year, up to a maximum of 25%, forfeited entirely when a claim is made.",
                            "numeric",
                            answer_keys=["5%", "25%", "five percent"],
                        )
                    ],
                    entities={"ncd_pct": 5, "ncd_max_pct": 25},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Sentinel Health Assurance Limited",
        product_name="Nirogam Plus Health Plan",
        product_type="Health Indemnity",
        uin="SHALHLIP24007V022425",
        version="1.4",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Policy 3 - Private car package
# --------------------------------------------------------------------------------------


def build_motor_package() -> Policy:
    pid = "MOT-DRIVESHIELD"
    sections = [
        Section(
            "1",
            "Preamble and Definitions",
            [
                Clause(
                    "1.1",
                    "Operative Clause",
                    "Meridian General Insurance Company Limited agrees, subject to the terms of this Policy, "
                    "to indemnify the Insured against loss of or damage to the Insured Vehicle and to cover "
                    "the Insured's legal liability to third parties arising out of the use of the Insured "
                    "Vehicle during the Policy Period.",
                ),
                Clause(
                    "1.2",
                    "Insured Declared Value",
                    "Insured Declared Value means the sum arrived at by applying the schedule of depreciation "
                    "set out in Clause 5.1 to the manufacturer's listed selling price of the vehicle and its "
                    "accessories. The Insured Declared Value shall be deemed to be the Sum Insured for the "
                    "purposes of Section I of this Policy and shall be the maximum amount payable in the event "
                    "of total loss or constructive total loss.",
                    qa=[
                        QA(
                            "What is Insured Declared Value under the DriveShield motor policy?",
                            "The manufacturer's listed selling price adjusted by the depreciation schedule; it is the sum insured for own damage and the maximum payable on total loss.",
                            "factual",
                            answer_keys=["idv", "listed selling price", "total loss"],
                        )
                    ],
                    entities={"definition": "Insured Declared Value"},
                ),
                Clause(
                    "1.3",
                    "Constructive Total Loss",
                    "The Insured Vehicle shall be treated as a constructive total loss where the aggregate "
                    "cost of retrieval and repair, calculated at the time of loss, exceeds seventy-five "
                    "percent (75%) of the Insured Declared Value.",
                    qa=[
                        QA(
                            "When is a vehicle a constructive total loss under the DriveShield policy?",
                            "When the cost of retrieval and repair exceeds 75% of the Insured Declared Value.",
                            "numeric",
                            answer_keys=["75%", "seventy-five"],
                        )
                    ],
                    entities={"definition": "Constructive Total Loss", "threshold_pct": 75},
                ),
            ],
        ),
        Section(
            "2",
            "Section I - Loss or Damage to the Insured Vehicle",
            [
                Clause(
                    "2.1",
                    "Perils Covered",
                    "The Company shall indemnify the Insured against loss or damage to the Insured Vehicle "
                    "caused by fire, explosion, self-ignition or lightning; burglary, housebreaking or theft; "
                    "riot and strike; earthquake, flood, inundation, storm, cyclone and hailstorm; accidental "
                    "external means; malicious act; terrorist activity; and whilst in transit by road, rail, "
                    "inland waterway, lift, elevator or air.",
                    qa=[
                        QA(
                            "Does the DriveShield motor policy cover flood damage?",
                            "Yes. Flood, inundation, storm, cyclone and hailstorm are listed perils under Section I own damage cover.",
                            "boolean",
                            answer_keys=["yes", "flood", "covered"],
                        )
                    ],
                ),
                Clause(
                    "2.2",
                    "Towing Charges",
                    "The Company shall pay the reasonable cost of removal of the Insured Vehicle to the "
                    "nearest repairer following an accident, and the cost of reasonable repairs necessary to "
                    "enable the vehicle to reach that repairer, up to Rs. 1,500 in respect of any one accident.",
                    qa=[
                        QA(
                            "What towing charges does the DriveShield policy pay after an accident?",
                            "Up to Rs. 1,500 per accident for removal of the vehicle to the nearest repairer.",
                            "numeric",
                            answer_keys=["1,500", "1500"],
                        )
                    ],
                    entities={"benefit": "Towing", "limit_inr": 1500},
                ),
                Clause(
                    "2.3",
                    "Consumables Exclusion under Own Damage",
                    "Unless the Consumables Cover add-on has been opted for and is shown in the Policy "
                    "Schedule, the cost of engine oil, gearbox oil, coolant, brake fluid, nut and bolt, screw, "
                    "washer, grease, lubricant, clip, air-conditioner gas, wheel balancing weight and "
                    "distilled water shall not be payable in a partial loss claim.",
                    qa=[
                        QA(
                            "Are consumables like engine oil covered in a DriveShield own damage claim?",
                            "No, not unless the Consumables Cover add-on has been opted for and appears in the policy schedule.",
                            "boolean",
                            answer_keys=["no", "add-on", "not payable"],
                        )
                    ],
                    entities={"exclusion": "Consumables"},
                ),
            ],
        ),
        Section(
            "3",
            "Section II - Liability to Third Parties",
            [
                Clause(
                    "3.1",
                    "Third Party Death and Bodily Injury",
                    "The Company shall indemnify the Insured against all sums which the Insured becomes "
                    "legally liable to pay in respect of the death of or bodily injury to any third party "
                    "caused by or arising out of the use of the Insured Vehicle. The liability of the Company "
                    "in respect of death or bodily injury is unlimited in amount.",
                    qa=[
                        QA(
                            "What is the third party death and bodily injury limit under the DriveShield policy?",
                            "Unlimited. The Company's liability for third party death or bodily injury is not capped in amount.",
                            "factual",
                            answer_keys=["unlimited", "no limit"],
                        )
                    ],
                    entities={"liability": "TP Bodily Injury", "limit": "unlimited"},
                ),
                Clause(
                    "3.2",
                    "Third Party Property Damage",
                    "The Company's liability in respect of damage to third party property shall not exceed "
                    "Rs. 7,50,000 in respect of any one accident, unless a higher limit has been opted for "
                    "and the corresponding premium has been paid.",
                    qa=[
                        QA(
                            "What is the third party property damage limit under the DriveShield policy?",
                            "Rs. 7,50,000 per accident, unless a higher limit has been opted for and paid for.",
                            "numeric",
                            answer_keys=["7,50,000", "750000", "7.5 lakh"],
                        )
                    ],
                    entities={"liability": "TP Property Damage", "limit_inr": 750000},
                ),
                Clause(
                    "3.3",
                    "Compulsory Personal Accident Cover for Owner-Driver",
                    "A compulsory personal accident cover of Rs. 15,00,000 is provided to the owner-driver "
                    "whilst driving the Insured Vehicle, including whilst mounting into, dismounting from or "
                    "travelling in the Insured Vehicle. The cover is subject to the owner-driver holding a "
                    "valid and effective driving licence at the time of the accident.",
                    qa=[
                        QA(
                            "What is the compulsory personal accident cover for the owner-driver under the DriveShield policy?",
                            "Rs. 15,00,000, subject to the owner-driver holding a valid and effective driving licence at the time of the accident.",
                            "numeric",
                            answer_keys=["15,00,000", "1500000", "15 lakh"],
                        )
                    ],
                    entities={"benefit": "Owner-Driver PA", "limit_inr": 1500000},
                ),
            ],
        ),
        Section(
            "4",
            "Add-On Covers",
            [
                Clause(
                    "4.1",
                    "Zero Depreciation Cover",
                    "Where this add-on has been opted for, the Company shall not apply depreciation to the "
                    "value of parts replaced in a partial loss claim, except in respect of tyres, tubes and "
                    "batteries, on which fifty percent (50%) depreciation shall continue to apply. This "
                    "add-on may be availed for a maximum of two (2) claims in a Policy Year.",
                    qa=[
                        QA(
                            "How many claims can I make under the DriveShield zero depreciation add-on in a year?",
                            "A maximum of 2 claims per policy year.",
                            "numeric",
                            answer_keys=["2", "two"],
                        ),
                        QA(
                            "Does zero depreciation on the DriveShield policy cover tyres?",
                            "No. Tyres, tubes and batteries continue to attract 50% depreciation even with zero depreciation cover.",
                            "boolean",
                            answer_keys=["no", "50%", "tyres"],
                        ),
                    ],
                    entities={"addon": "Zero Depreciation", "max_claims": 2, "tyre_dep_pct": 50},
                ),
                Clause(
                    "4.2",
                    "Engine Protect Cover",
                    "Where this add-on has been opted for, the Company shall indemnify the cost of repair or "
                    "replacement of the engine, gearbox and differential assembly consequent upon ingress of "
                    "water into the engine or leakage of lubricating oil, which would otherwise be excluded "
                    "as consequential loss. This cover does not apply where the Insured attempted to start "
                    "the vehicle after it was submerged in water.",
                    qa=[
                        QA(
                            "Does the DriveShield engine protect add-on cover water ingress?",
                            "Yes, but not where the insured attempted to start the vehicle after it was submerged.",
                            "boolean",
                            answer_keys=["yes", "water ingress", "not", "submerged"],
                        )
                    ],
                    entities={"addon": "Engine Protect"},
                ),
                Clause(
                    "4.3",
                    "Return to Invoice Cover",
                    "In the event of total loss, constructive total loss or theft of the Insured Vehicle, the "
                    "Company shall pay the difference between the Insured Declared Value and the current "
                    "invoice value of a new vehicle of the same make, model and variant, together with the "
                    "first-time registration charges and road tax. This add-on is available only for vehicles "
                    "up to five (5) years of age.",
                    qa=[
                        QA(
                            "What is the vehicle age limit for the DriveShield return to invoice add-on?",
                            "5 years. The add-on is available only for vehicles up to five years of age.",
                            "numeric",
                            answer_keys=["5 years", "five"],
                        )
                    ],
                    entities={"addon": "Return to Invoice", "max_vehicle_age_years": 5},
                ),
                Clause(
                    "4.4",
                    "Roadside Assistance",
                    "Where this add-on has been opted for, the Company shall arrange on-site minor repairs, "
                    "flat tyre assistance, battery jump start, emergency fuel delivery up to five (5) litres, "
                    "key retrieval and towing, available at any location within one hundred (100) kilometres "
                    "of the Insured's registered address. Availing this service shall not affect the No Claim "
                    "Bonus.",
                    qa=[
                        QA(
                            "Does using roadside assistance affect the no claim bonus on the DriveShield policy?",
                            "No. Availing roadside assistance does not affect the No Claim Bonus.",
                            "boolean",
                            answer_keys=["no", "not affect", "ncb"],
                        )
                    ],
                    entities={"addon": "Roadside Assistance", "radius_km": 100, "fuel_litres": 5},
                ),
            ],
        ),
        Section(
            "5",
            "Depreciation and Deductibles",
            [
                Clause(
                    "5.1",
                    "Depreciation Schedule for Parts",
                    "In a partial loss claim, depreciation on the value of parts replaced shall be applied as "
                    "follows: for all rubber, nylon and plastic parts, tyres, tubes, batteries and air bags, "
                    "fifty percent (50%); for all parts made of fibre glass, thirty percent (30%); for all "
                    "parts made of glass, nil; and for all other parts, depreciation shall be applied "
                    "according to the age of the vehicle as set out in Clause 5.2.",
                    qa=[
                        QA(
                            "What depreciation applies to plastic parts in a DriveShield claim?",
                            "50% depreciation applies to rubber, nylon and plastic parts, tyres, tubes, batteries and air bags.",
                            "numeric",
                            answer_keys=["50%", "fifty percent"],
                        ),
                        QA(
                            "What depreciation applies to glass parts under the DriveShield policy?",
                            "Nil. No depreciation is applied to parts made of glass.",
                            "numeric",
                            answer_keys=["nil", "no depreciation", "zero"],
                        ),
                    ],
                    entities={"dep_plastic_pct": 50, "dep_fibre_pct": 30, "dep_glass_pct": 0},
                ),
                Clause(
                    "5.2",
                    "Age-Based Depreciation on Metal Parts",
                    "For metal parts, depreciation shall be nil where the vehicle is not exceeding six (6) "
                    "months old; five percent (5%) where exceeding six months but not exceeding one year; "
                    "ten percent (10%) exceeding one year but not exceeding two years; fifteen percent (15%) "
                    "exceeding two years but not exceeding three years; twenty-five percent (25%) exceeding "
                    "three years but not exceeding four years; thirty-five percent (35%) exceeding four years "
                    "but not exceeding five years; and fifty percent (50%) exceeding five years.",
                    qa=[
                        QA(
                            "What depreciation applies to metal parts on a three-and-a-half year old car under the DriveShield policy?",
                            "25%. Vehicles exceeding three years but not exceeding four years attract 25% depreciation on metal parts.",
                            "numeric",
                            answer_keys=["25%", "twenty-five"],
                        )
                    ],
                    entities={"dep_schedule": "age-based metal"},
                ),
                Clause(
                    "5.3",
                    "Compulsory Deductible",
                    "A compulsory deductible of Rs. 1,000 shall apply to each and every own damage claim in "
                    "respect of private cars with an engine capacity not exceeding 1500 cc, and Rs. 2,000 "
                    "where the engine capacity exceeds 1500 cc. Any voluntary deductible opted for shall apply "
                    "in addition to the compulsory deductible.",
                    qa=[
                        QA(
                            "What is the compulsory deductible under the DriveShield policy?",
                            "Rs. 1,000 for private cars up to 1500 cc and Rs. 2,000 above 1500 cc, in addition to any voluntary deductible.",
                            "numeric",
                            answer_keys=["1,000", "2,000", "1500 cc"],
                        )
                    ],
                    entities={"deductible_small_inr": 1000, "deductible_large_inr": 2000, "cc_split": 1500},
                ),
            ],
        ),
        Section(
            "6",
            "Exclusions",
            [
                Clause(
                    "6.1",
                    "General Exclusions",
                    "The Company shall not be liable for consequential loss, depreciation, wear and tear, "
                    "mechanical or electrical breakdown, failures or breakages, damage to tyres and tubes "
                    "unless the Insured Vehicle is damaged at the same time, or loss or damage arising whilst "
                    "the vehicle is used outside the limitations as to use stated in the Policy Schedule.",
                    qa=[
                        QA(
                            "Is mechanical breakdown covered under the DriveShield policy?",
                            "No. Mechanical and electrical breakdown, failures and breakages are excluded.",
                            "boolean",
                            answer_keys=["no", "excluded", "breakdown"],
                        )
                    ],
                    entities={"exclusion": "Mechanical Breakdown"},
                ),
                Clause(
                    "6.2",
                    "Driving Without a Valid Licence",
                    "No claim shall be payable where, at the time of the accident, the vehicle was being "
                    "driven by a person not holding a valid and effective driving licence for the class of "
                    "vehicle concerned, or by a person who was disqualified from holding such a licence.",
                    qa=[
                        QA(
                            "Will the DriveShield policy pay if the driver had no valid licence?",
                            "No. Claims are not payable where the driver did not hold a valid and effective licence for that class of vehicle, or was disqualified.",
                            "boolean",
                            answer_keys=["no", "not payable", "licence"],
                        )
                    ],
                    entities={"exclusion": "Invalid Licence"},
                ),
                Clause(
                    "6.3",
                    "Driving Under the Influence",
                    "No claim shall be payable where the person driving the Insured Vehicle at the time of "
                    "the accident was under the influence of intoxicating liquor or drugs.",
                    entities={"exclusion": "Intoxication"},
                ),
            ],
        ),
        Section(
            "7",
            "Claims and Conditions",
            [
                Clause(
                    "7.1",
                    "Notice of Accident",
                    "Notice of any accident, loss or damage shall be given to the Company in writing "
                    "immediately, and in any event within forty-eight (48) hours of the occurrence. In the "
                    "case of theft or criminal act, the Insured shall also lodge a First Information Report "
                    "with the police without delay and provide a copy to the Company.",
                    qa=[
                        QA(
                            "Within how long must an accident be reported under the DriveShield policy?",
                            "Immediately, and in any event within 48 hours of the occurrence; theft also requires a police First Information Report.",
                            "numeric",
                            answer_keys=["48 hours", "forty-eight"],
                        )
                    ],
                    entities={"accident_notice_hours": 48},
                ),
                Clause(
                    "7.2",
                    "No Claim Bonus",
                    "Where no claim has been made or pending during the preceding Policy Period, a No Claim "
                    "Bonus shall be allowed on the own damage premium at renewal as follows: twenty percent "
                    "(20%) after one claim-free year, twenty-five percent (25%) after two consecutive "
                    "claim-free years, thirty-five percent (35%) after three, forty-five percent (45%) after "
                    "four, and fifty percent (50%) after five consecutive claim-free years.",
                    qa=[
                        QA(
                            "What is the maximum no claim bonus under the DriveShield motor policy?",
                            "50%, reached after five consecutive claim-free years.",
                            "numeric",
                            answer_keys=["50%", "fifty percent", "five"],
                        ),
                        QA(
                            "What no claim bonus applies after two consecutive claim-free years on the DriveShield policy?",
                            "25% on the own damage premium.",
                            "numeric",
                            answer_keys=["25%", "twenty-five"],
                        ),
                    ],
                    entities={"ncb_year1": 20, "ncb_year5": 50},
                ),
                Clause(
                    "7.3",
                    "Transfer of Interest",
                    "On transfer of ownership of the Insured Vehicle, the benefit of this Policy shall not "
                    "pass to the transferee unless the Company's consent in writing is obtained within "
                    "fourteen (14) days of the date of transfer. The No Claim Bonus earned by the previous "
                    "owner shall not transfer and remains to the credit of that owner.",
                    qa=[
                        QA(
                            "Does the no claim bonus transfer when I sell my car insured under the DriveShield policy?",
                            "No. The no claim bonus stays with the previous owner; policy transfer itself requires the Company's written consent within 14 days.",
                            "boolean",
                            answer_keys=["no", "14 days", "previous owner"],
                        )
                    ],
                    entities={"transfer_consent_days": 14},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Meridian General Insurance Company Limited",
        product_name="DriveShield Private Car Package Policy",
        product_type="Motor Private Car",
        uin="MGICMOTP24019V012425",
        version="3.0",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Policy 4 - Term life
# --------------------------------------------------------------------------------------


def build_term_life() -> Policy:
    pid = "LIF-SURAKSHAKAVACH"
    sections = [
        Section(
            "1",
            "Preamble",
            [
                Clause(
                    "1.1",
                    "Nature of Contract",
                    "Suraksha Kavach Term Assurance is a non-linked, non-participating, pure risk premium "
                    "individual life insurance plan. It acquires no surrender value, no paid-up value and no "
                    "maturity value. The Death Benefit is payable only on the death of the Life Assured during "
                    "the Policy Term whilst the Policy is in force.",
                    qa=[
                        QA(
                            "Does the Suraksha Kavach term plan have a maturity value?",
                            "No. It is a pure risk plan with no surrender value, no paid-up value and no maturity value.",
                            "boolean",
                            answer_keys=["no", "no maturity", "pure risk"],
                        )
                    ],
                ),
                Clause(
                    "1.2",
                    "Eligibility",
                    "The minimum age at entry is eighteen (18) years and the maximum age at entry is "
                    "sixty-five (65) years, both as on last birthday. The maximum maturity age is eighty-five "
                    "(85) years. The minimum Sum Assured is Rs. 25,00,000 and the Policy Term ranges from "
                    "ten (10) to forty (40) years.",
                    qa=[
                        QA(
                            "What is the maximum entry age for the Suraksha Kavach term plan?",
                            "65 years as on last birthday, with a maximum maturity age of 85 years.",
                            "numeric",
                            answer_keys=["65", "sixty-five"],
                        ),
                        QA(
                            "What is the minimum sum assured under the Suraksha Kavach term plan?",
                            "Rs. 25,00,000.",
                            "numeric",
                            answer_keys=["25,00,000", "2500000", "25 lakh"],
                        ),
                    ],
                    entities={
                        "min_entry_age": 18,
                        "max_entry_age": 65,
                        "max_maturity_age": 85,
                        "min_sum_assured_inr": 2500000,
                    },
                ),
            ],
        ),
        Section(
            "2",
            "Benefits",
            [
                Clause(
                    "2.1",
                    "Death Benefit",
                    "On the death of the Life Assured during the Policy Term, provided the Policy is in force, "
                    "the Death Benefit payable shall be the highest of ten (10) times the annualised premium, "
                    "one hundred and five percent (105%) of all premiums paid up to the date of death, or the "
                    "absolute Sum Assured stated in the Policy Schedule.",
                    qa=[
                        QA(
                            "How is the death benefit calculated under the Suraksha Kavach term plan?",
                            "The highest of 10 times the annualised premium, 105% of all premiums paid to date of death, or the absolute sum assured.",
                            "factual",
                            answer_keys=["10 times", "105%", "sum assured"],
                        )
                    ],
                    entities={"benefit": "Death Benefit", "multiple": 10, "premium_pct": 105},
                ),
                Clause(
                    "2.2",
                    "Terminal Illness Benefit",
                    "Where the Life Assured is diagnosed with a Terminal Illness and is certified by two "
                    "independent Medical Practitioners as having a life expectancy not exceeding six (6) "
                    "months, the Company shall accelerate payment of fifty percent (50%) of the Sum Assured, "
                    "subject to a maximum of Rs. 1,00,00,000. The Sum Assured payable on subsequent death "
                    "shall stand reduced by the amount so accelerated.",
                    qa=[
                        QA(
                            "What does the terminal illness benefit pay under the Suraksha Kavach term plan?",
                            "50% of the sum assured accelerated, capped at Rs. 1,00,00,000, where life expectancy is certified at six months or less by two independent medical practitioners.",
                            "numeric",
                            answer_keys=["50%", "1,00,00,000", "6 months"],
                        )
                    ],
                    entities={"benefit": "Terminal Illness", "accelerate_pct": 50, "cap_inr": 10000000},
                ),
                Clause(
                    "2.3",
                    "Death Benefit Payout Options",
                    "The Nominee may elect to receive the Death Benefit as a lump sum, as a level monthly "
                    "income over sixty (60), one hundred and twenty (120) or one hundred and eighty (180) "
                    "months, or as a combination of a lump sum of not less than twenty percent (20%) of the "
                    "Death Benefit with the balance paid as monthly income.",
                    qa=[
                        QA(
                            "What payout options are available to the nominee under the Suraksha Kavach term plan?",
                            "A lump sum, a level monthly income over 60, 120 or 180 months, or a combination with a lump sum of at least 20% and the balance as monthly income.",
                            "factual",
                            answer_keys=["lump sum", "monthly income", "60", "120", "180"],
                        )
                    ],
                ),
            ],
        ),
        Section(
            "3",
            "Riders",
            [
                Clause(
                    "3.1",
                    "Accidental Death Benefit Rider",
                    "Where this rider has been opted for, an additional amount equal to the Rider Sum Assured "
                    "shall be payable if the death of the Life Assured occurs as a direct result of an "
                    "Accident, provided death occurs within one hundred and eighty (180) days of the Accident. "
                    "The Rider Sum Assured shall not exceed the base Sum Assured.",
                    qa=[
                        QA(
                            "Within what period must death occur for the accidental death rider to pay under the Suraksha Kavach plan?",
                            "Within 180 days of the accident.",
                            "numeric",
                            answer_keys=["180 days", "one hundred and eighty"],
                        )
                    ],
                    entities={"rider": "Accidental Death", "window_days": 180},
                ),
                Clause(
                    "3.2",
                    "Critical Illness Rider",
                    "Where this rider has been opted for, a lump sum equal to the Rider Sum Assured shall be "
                    "payable on first diagnosis of any one of the thirty-four (34) listed Critical Illnesses, "
                    "subject to a survival period of thirty (30) days from the date of diagnosis and a waiting "
                    "period of ninety (90) days from the rider commencement date.",
                    qa=[
                        QA(
                            "What is the survival period under the Suraksha Kavach critical illness rider?",
                            "30 days from the date of diagnosis, with a 90-day waiting period from rider commencement.",
                            "numeric",
                            answer_keys=["30 days", "90 days", "thirty"],
                        )
                    ],
                    entities={"rider": "Critical Illness", "survival_days": 30, "waiting_days": 90, "count": 34},
                ),
            ],
        ),
        Section(
            "4",
            "Exclusions",
            [
                Clause(
                    "4.1",
                    "Suicide Clause",
                    "In the event of death of the Life Assured due to suicide within twelve (12) months from "
                    "the date of commencement of risk or from the date of revival of the Policy, whichever is "
                    "later, the Nominee shall be entitled to at least eighty percent (80%) of the total "
                    "premiums paid up to the date of death, or the surrender value available as at the date "
                    "of death, whichever is higher, provided the Policy is in force.",
                    qa=[
                        QA(
                            "What does the Suraksha Kavach term plan pay if death occurs within 12 months of commencement in excluded circumstances under clause 4.1?",
                            "At least 80% of total premiums paid up to the date of death, or the surrender value then available, whichever is higher.",
                            "numeric",
                            answer_keys=["80%", "eighty percent", "premiums paid"],
                        )
                    ],
                    entities={"exclusion": "Suicide", "months": 12, "refund_pct": 80},
                ),
                Clause(
                    "4.2",
                    "Rider Exclusions",
                    "No rider benefit shall be payable where the event arises from the Life Assured's "
                    "participation in any criminal act with unlawful intent, service in the armed forces "
                    "during a period of war or civil commotion, participation in aviation other than as a "
                    "fare-paying passenger on a licensed commercial airline, or participation in hazardous "
                    "sports or racing of any kind.",
                    qa=[
                        QA(
                            "Does the Suraksha Kavach accidental death rider cover private flying?",
                            "No. Aviation other than travel as a fare-paying passenger on a licensed commercial airline is excluded from rider benefits.",
                            "boolean",
                            answer_keys=["no", "excluded", "fare-paying"],
                        )
                    ],
                    entities={"exclusion": "Rider Exclusions"},
                ),
            ],
        ),
        Section(
            "5",
            "Premium, Grace, Lapse and Revival",
            [
                Clause(
                    "5.1",
                    "Grace Period",
                    "A grace period of thirty (30) days is allowed for payment of premium where the premium "
                    "payment mode is yearly, half-yearly or quarterly, and fifteen (15) days where the mode "
                    "is monthly. The Policy remains in force during the grace period.",
                    qa=[
                        QA(
                            "What is the grace period under the Suraksha Kavach term plan?",
                            "30 days for yearly, half-yearly and quarterly modes, and 15 days for monthly mode.",
                            "numeric",
                            answer_keys=["30 days", "15 days", "thirty"],
                        )
                    ],
                    entities={"grace_days": 30, "grace_days_monthly": 15},
                ),
                Clause(
                    "5.2",
                    "Lapse",
                    "If premium is not paid before the expiry of the grace period, the Policy shall lapse "
                    "from the due date of the first unpaid premium and no benefit shall be payable thereafter, "
                    "as this Policy acquires no surrender value.",
                ),
                Clause(
                    "5.3",
                    "Revival",
                    "A lapsed Policy may be revived within five (5) years from the date of the first unpaid "
                    "premium, subject to payment of all arrears of premium together with interest at the rate "
                    "declared by the Company, and to the Life Assured furnishing satisfactory evidence of "
                    "continued insurability at their own cost.",
                    qa=[
                        QA(
                            "How long do I have to revive a lapsed Suraksha Kavach policy?",
                            "5 years from the date of the first unpaid premium, subject to arrears with interest and satisfactory evidence of insurability.",
                            "numeric",
                            answer_keys=["5 years", "five"],
                        )
                    ],
                    entities={"revival_years": 5},
                ),
                Clause(
                    "5.4",
                    "Free Look Period",
                    "The Policyholder may return the Policy within thirty (30) days of receipt of the Policy "
                    "document if dissatisfied with its terms, and shall receive a refund of premium paid less "
                    "proportionate risk premium for the period of cover, medical examination expenses and "
                    "stamp duty charges.",
                    qa=[
                        QA(
                            "What is the free look period under the Suraksha Kavach term plan?",
                            "30 days from receipt of the policy document.",
                            "numeric",
                            answer_keys=["30 days", "thirty"],
                        )
                    ],
                    entities={"free_look_days": 30},
                ),
            ],
        ),
        Section(
            "6",
            "Claims and General Provisions",
            [
                Clause(
                    "6.1",
                    "Claim Intimation and Documents",
                    "A death claim shall be intimated to the Company at the earliest and in any event within "
                    "ninety (90) days of the date of death. The Nominee shall submit the original Policy "
                    "document, the death certificate, the claimant's statement, proof of identity, and where "
                    "applicable the post-mortem report and police First Information Report.",
                    qa=[
                        QA(
                            "Within how many days must a death claim be intimated under the Suraksha Kavach plan?",
                            "Within 90 days of the date of death.",
                            "numeric",
                            answer_keys=["90 days", "ninety"],
                        )
                    ],
                    entities={"claim_intimation_days": 90},
                ),
                Clause(
                    "6.2",
                    "Claim Settlement Timeline",
                    "Where no investigation is warranted, the Company shall settle a death claim within "
                    "fifteen (15) days of receipt of all required documents. Where an investigation is "
                    "required, it shall be completed within ninety (90) days of receipt of the claim "
                    "intimation and the claim settled within thirty (30) days thereafter.",
                    qa=[
                        QA(
                            "How quickly is a death claim settled under the Suraksha Kavach plan?",
                            "Within 15 days of receiving all documents where no investigation is needed; where investigation is required it completes within 90 days and settlement follows within 30 days.",
                            "numeric",
                            answer_keys=["15 days", "90 days", "30 days"],
                        )
                    ],
                    entities={"settle_days": 15, "investigation_days": 90},
                ),
                Clause(
                    "6.3",
                    "Non-Disclosure and Section 45",
                    "No Policy shall be called in question on the ground of misstatement or suppression of "
                    "material fact after the expiry of three (3) years from the date of issuance of the "
                    "Policy, the date of commencement of risk, the date of revival, or the date of the rider "
                    "to the Policy, whichever is later, in accordance with Section 45 of the Insurance Act.",
                    qa=[
                        QA(
                            "After how long can the Suraksha Kavach policy no longer be questioned for non-disclosure?",
                            "3 years from issuance, commencement of risk, revival or the rider date, whichever is later, under Section 45 of the Insurance Act.",
                            "numeric",
                            answer_keys=["3 years", "three", "section 45"],
                        )
                    ],
                    entities={"section_45_years": 3},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Sentinel Life Assurance Limited",
        product_name="Suraksha Kavach Term Assurance",
        product_type="Term Life",
        uin="SLALTERM24003V012425",
        version="1.2",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Policy 5 - Householder's package
# --------------------------------------------------------------------------------------


def build_home_package() -> Policy:
    pid = "HOM-GRIHARAKSHA"
    sections = [
        Section(
            "1",
            "Preamble and Definitions",
            [
                Clause(
                    "1.1",
                    "Operative Clause",
                    "Meridian General Insurance Company Limited agrees to indemnify the Insured in respect of "
                    "loss of or damage to the Insured Property described in the Policy Schedule occurring "
                    "during the Policy Period, subject to the Sums Insured, terms, conditions and exclusions "
                    "of each Section opted for.",
                ),
                Clause(
                    "1.2",
                    "Reinstatement Value",
                    "Reinstatement Value means the cost of replacing or reinstating property of the same kind "
                    "or type by new property, without deduction for depreciation. Building cover under Section "
                    "A operates on a Reinstatement Value basis, whilst Contents cover under Section B operates "
                    "on an Indemnity basis with depreciation applied, unless the Reinstatement Value option "
                    "has been opted for.",
                    qa=[
                        QA(
                            "Is contents cover under the Griha Raksha policy on a reinstatement or indemnity basis?",
                            "Indemnity basis with depreciation applied, unless the reinstatement value option has been opted for. Building cover is on reinstatement value.",
                            "factual",
                            answer_keys=["indemnity", "depreciation", "reinstatement"],
                        )
                    ],
                    entities={"definition": "Reinstatement Value"},
                ),
            ],
        ),
        Section(
            "2",
            "Section A - Building",
            [
                Clause(
                    "2.1",
                    "Perils Covered",
                    "The Company shall indemnify loss or damage to the building caused by fire, lightning, "
                    "explosion or implosion, aircraft damage, riot, strike and malicious damage, storm, "
                    "cyclone, flood and inundation, impact damage by any rail or road vehicle, subsidence and "
                    "landslide including rockslide, bursting or overflowing of water tanks and pipes, missile "
                    "testing operations, leakage from automatic sprinkler installations, and bush fire.",
                    qa=[
                        QA(
                            "Does the Griha Raksha policy cover damage from bursting water pipes?",
                            "Yes. Bursting or overflowing of water tanks, apparatus and pipes is a covered peril under Section A building cover.",
                            "boolean",
                            answer_keys=["yes", "covered", "bursting"],
                        )
                    ],
                ),
                Clause(
                    "2.2",
                    "Architects and Surveyors Fees",
                    "Fees of architects, surveyors and consulting engineers necessarily incurred in the "
                    "reinstatement of the building following an admitted claim shall be payable up to three "
                    "percent (3%) of the claim amount.",
                    qa=[
                        QA(
                            "Are architect fees covered under the Griha Raksha policy?",
                            "Yes, up to 3% of the claim amount where necessarily incurred in reinstating the building after an admitted claim.",
                            "numeric",
                            answer_keys=["3%", "three percent"],
                        )
                    ],
                    entities={"benefit": "Architect Fees", "limit_pct": 3},
                ),
                Clause(
                    "2.3",
                    "Debris Removal",
                    "The reasonable cost of removal of debris from the insured premises following an admitted "
                    "claim shall be payable up to two percent (2%) of the claim amount.",
                    entities={"benefit": "Debris Removal", "limit_pct": 2},
                ),
                Clause(
                    "2.4",
                    "Loss of Rent and Alternative Accommodation",
                    "Where the insured building is rendered uninhabitable following an admitted claim, the "
                    "Company shall pay the rent for alternative accommodation of a similar standard for the "
                    "period reasonably required for reinstatement, up to a maximum of twelve (12) months and "
                    "not exceeding ten percent (10%) of the Sum Insured on the building.",
                    qa=[
                        QA(
                            "Does the Griha Raksha policy pay for alternative accommodation?",
                            "Yes, for up to 12 months and not exceeding 10% of the building sum insured, where the home is rendered uninhabitable after an admitted claim.",
                            "boolean",
                            answer_keys=["yes", "12 months", "10%"],
                        )
                    ],
                    entities={"benefit": "Alternative Accommodation", "max_months": 12, "limit_pct_si": 10},
                ),
            ],
        ),
        Section(
            "3",
            "Section B - Contents",
            [
                Clause(
                    "3.1",
                    "Scope",
                    "Household contents, furniture, fixtures, fittings, domestic appliances and personal "
                    "effects belonging to the Insured or to any member of the Insured's family permanently "
                    "residing with them are covered against the perils listed in Clause 2.1, up to the Sum "
                    "Insured for Section B.",
                ),
                Clause(
                    "3.2",
                    "Valuables Sub-Limit",
                    "Jewellery, precious stones, articles of gold and silver, watches, curios and works of "
                    "art are covered only up to twenty-five percent (25%) of the Section B Sum Insured, or "
                    "Rs. 2,00,000, whichever is lower, unless specifically declared and listed in the Policy "
                    "Schedule with valuation certificates.",
                    qa=[
                        QA(
                            "What is the jewellery sub-limit under the Griha Raksha policy?",
                            "25% of the contents sum insured or Rs. 2,00,000, whichever is lower, unless items are specifically declared with valuation certificates.",
                            "numeric",
                            answer_keys=["25%", "2,00,000", "200000"],
                        )
                    ],
                    entities={"sub_limit": "Valuables", "pct_si": 25, "cap_inr": 200000},
                ),
                Clause(
                    "3.3",
                    "Single Article Limit",
                    "In respect of contents that have not been specifically declared, the liability of the "
                    "Company for any one article shall not exceed five percent (5%) of the Section B Sum "
                    "Insured.",
                    entities={"sub_limit": "Single Article", "pct_si": 5},
                ),
            ],
        ),
        Section(
            "4",
            "Section C - Burglary and Theft",
            [
                Clause(
                    "4.1",
                    "Scope",
                    "Loss of or damage to contents by burglary, housebreaking or theft following actual "
                    "forcible and violent entry into or exit from the insured premises, including damage to "
                    "the premises caused in the course of such entry, is covered up to the Section C Sum "
                    "Insured.",
                    qa=[
                        QA(
                            "Does the Griha Raksha burglary section cover theft without forced entry?",
                            "No. Cover requires actual forcible and violent entry into or exit from the insured premises.",
                            "boolean",
                            answer_keys=["no", "forcible", "violent"],
                        )
                    ],
                ),
                Clause(
                    "4.2",
                    "Unoccupancy Condition",
                    "Cover under Section C shall be suspended where the insured premises remain unoccupied "
                    "for a continuous period exceeding thirty (30) days, unless prior written notice has been "
                    "given to the Company and the Company has agreed in writing to continue cover.",
                    qa=[
                        QA(
                            "What happens to burglary cover if my home is empty for a long period under the Griha Raksha policy?",
                            "Cover is suspended once the premises are unoccupied for more than 30 continuous days, unless the Company has agreed in writing to continue cover.",
                            "numeric",
                            answer_keys=["30 days", "thirty", "suspended"],
                        )
                    ],
                    entities={"condition": "Unoccupancy", "days": 30},
                ),
            ],
        ),
        Section(
            "5",
            "Section D - Portable Electronic Equipment",
            [
                Clause(
                    "5.1",
                    "Scope and Territorial Limit",
                    "Portable electronic equipment including laptops, tablets, cameras and mobile telephones "
                    "belonging to the Insured is covered against accidental damage and theft anywhere in "
                    "India, and worldwide for a maximum of sixty (60) days in any Policy Year whilst the "
                    "Insured is travelling.",
                    qa=[
                        QA(
                            "Is my laptop covered abroad under the Griha Raksha policy?",
                            "Yes, worldwide for a maximum of 60 days in any policy year whilst travelling, and anywhere in India otherwise.",
                            "boolean",
                            answer_keys=["yes", "60 days", "worldwide"],
                        )
                    ],
                    entities={"benefit": "Portable Electronics", "worldwide_days": 60},
                ),
                Clause(
                    "5.2",
                    "Depreciation on Portable Equipment",
                    "In the event of a total loss of portable electronic equipment, depreciation shall be "
                    "applied at ten percent (10%) per completed year from the date of purchase, subject to a "
                    "maximum of fifty percent (50%). Theft from an unattended vehicle is not covered unless "
                    "the equipment was in a locked boot and there is evidence of forcible entry.",
                    qa=[
                        QA(
                            "What depreciation applies to a stolen laptop under the Griha Raksha policy?",
                            "10% per completed year from purchase, capped at 50%.",
                            "numeric",
                            answer_keys=["10%", "50%", "ten percent"],
                        ),
                        QA(
                            "Is a laptop stolen from a car covered under the Griha Raksha policy?",
                            "Only if it was in a locked boot and there is evidence of forcible entry; theft from an unattended vehicle is otherwise not covered.",
                            "boolean",
                            answer_keys=["locked boot", "forcible", "not covered"],
                        ),
                    ],
                    entities={"dep_pct_per_year": 10, "dep_max_pct": 50},
                ),
            ],
        ),
        Section(
            "6",
            "Exclusions and Conditions",
            [
                Clause(
                    "6.1",
                    "General Exclusions",
                    "This Policy does not cover loss or damage caused by wear and tear, gradual deterioration, "
                    "moth, vermin, insects, rust, mildew or dampness; loss of cash, bank notes, securities or "
                    "documents unless specifically covered; property held in trust or on commission unless "
                    "declared; or loss arising from war, nuclear perils or wilful act of the Insured.",
                    qa=[
                        QA(
                            "Is cash covered under the Griha Raksha policy?",
                            "No. Loss of cash, bank notes, securities and documents is excluded unless specifically covered.",
                            "boolean",
                            answer_keys=["no", "excluded", "unless specifically"],
                        )
                    ],
                    entities={"exclusion": "General"},
                ),
                Clause(
                    "6.2",
                    "Underinsurance and Average",
                    "Where at the time of loss the Sum Insured is less than the value of the insured property, "
                    "the Insured shall be considered their own insurer for the difference and shall bear a "
                    "rateable proportion of the loss. This condition of average applies separately to each "
                    "Section and to each item of the Policy Schedule.",
                    qa=[
                        QA(
                            "What happens if my Griha Raksha sum insured is too low at the time of loss?",
                            "The condition of average applies: the insured bears a rateable proportion of the loss, separately for each section and item.",
                            "procedural",
                            answer_keys=["average", "rateable", "proportion"],
                        )
                    ],
                    entities={"condition": "Average"},
                ),
                Clause(
                    "6.3",
                    "Claim Notification",
                    "The Insured shall notify the Company of any event likely to give rise to a claim within "
                    "seven (7) days of its occurrence, and in the case of burglary, theft or malicious damage, "
                    "shall also lodge a complaint with the police within twenty-four (24) hours.",
                    qa=[
                        QA(
                            "Within how long must I report a claim under the Griha Raksha policy?",
                            "Within 7 days of the event, and burglary, theft or malicious damage must also be reported to the police within 24 hours.",
                            "numeric",
                            answer_keys=["7 days", "seven", "24 hours"],
                        )
                    ],
                    entities={"notify_days": 7, "police_hours": 24},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Meridian General Insurance Company Limited",
        product_name="Griha Raksha Householder's Package Policy",
        product_type="Home / Householder",
        uin="MGICHOMP24011V012425",
        version="2.0",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Policy 6 - Overseas travel
# --------------------------------------------------------------------------------------


def build_travel() -> Policy:
    pid = "TRV-YATRASAFE"
    sections = [
        Section(
            "1",
            "Preamble and Definitions",
            [
                Clause(
                    "1.1",
                    "Scope of Contract",
                    "Sentinel Health Assurance Limited agrees to provide the benefits described in this Policy "
                    "in respect of events occurring during a Trip undertaken by the Insured Person outside the "
                    "Republic of India within the Policy Period, subject to the limits stated in the Schedule "
                    "of Benefits.",
                ),
                Clause(
                    "1.2",
                    "Trip",
                    "Trip means a journey commencing when the Insured Person crosses the Indian immigration "
                    "outbound counter and ending on the earlier of the Insured Person clearing Indian "
                    "immigration on return, or the expiry of the Policy Period. The maximum duration of any "
                    "single Trip shall be one hundred and eighty (180) days.",
                    qa=[
                        QA(
                            "What is the maximum single trip duration under the Yatra Safe travel policy?",
                            "180 days.",
                            "numeric",
                            answer_keys=["180 days", "one hundred and eighty"],
                        )
                    ],
                    entities={"definition": "Trip", "max_days": 180},
                ),
            ],
        ),
        Section(
            "2",
            "Medical Benefits",
            [
                Clause(
                    "2.1",
                    "Emergency Medical Expenses",
                    "The Company shall indemnify expenses necessarily incurred outside India on emergency "
                    "medical treatment of an Illness contracted or Injury sustained during the Trip, up to "
                    "the Sum Insured stated in the Schedule, subject to a deductible of USD 100 per claim. "
                    "Treatment of a condition known to exist before the Trip commenced is not covered save "
                    "as provided in Clause 2.3.",
                    qa=[
                        QA(
                            "What is the deductible on emergency medical expenses under the Yatra Safe policy?",
                            "USD 100 per claim.",
                            "numeric",
                            answer_keys=["usd 100", "100"],
                        )
                    ],
                    entities={"benefit": "Emergency Medical", "deductible_usd": 100},
                ),
                Clause(
                    "2.2",
                    "Emergency Medical Evacuation and Repatriation",
                    "Where the attending Medical Practitioner and the Company's assistance provider jointly "
                    "determine that the Insured Person must be evacuated to the nearest facility capable of "
                    "providing adequate care, or repatriated to India, the Company shall arrange and pay for "
                    "such evacuation or repatriation up to USD 50,000. Repatriation of mortal remains to "
                    "India is covered up to USD 10,000.",
                    qa=[
                        QA(
                            "What is the medical evacuation limit under the Yatra Safe policy?",
                            "USD 50,000 for emergency evacuation or repatriation, and USD 10,000 for repatriation of mortal remains.",
                            "numeric",
                            answer_keys=["50,000", "50000", "10,000"],
                        )
                    ],
                    entities={"benefit": "Evacuation", "limit_usd": 50000, "remains_usd": 10000},
                ),
                Clause(
                    "2.3",
                    "Pre-Existing Condition in Life-Threatening Emergency",
                    "Notwithstanding Clause 2.1, expenses incurred on immediate relief of a Pre-Existing "
                    "Condition which manifests as a life-threatening emergency during the Trip shall be "
                    "payable up to USD 10,000, limited to such treatment as is necessary to stabilise the "
                    "Insured Person for return to India.",
                    qa=[
                        QA(
                            "Does the Yatra Safe policy cover a pre-existing condition emergency abroad?",
                            "Yes, up to USD 10,000, but only for treatment necessary to stabilise the insured for return to India.",
                            "boolean",
                            answer_keys=["usd 10,000", "10,000", "stabilise", "yes"],
                        )
                    ],
                    entities={"benefit": "PED Emergency", "limit_usd": 10000},
                ),
                Clause(
                    "2.4",
                    "Dental Treatment Abroad",
                    "Emergency dental treatment necessitated by acute pain or by an Accident during the Trip "
                    "shall be payable up to USD 500. Routine dental care, crowns, bridges and orthodontic "
                    "work are excluded.",
                    entities={"benefit": "Emergency Dental", "limit_usd": 500},
                ),
            ],
        ),
        Section(
            "3",
            "Travel Inconvenience Benefits",
            [
                Clause(
                    "3.1",
                    "Trip Cancellation",
                    "Where the Trip is cancelled before departure owing to the death, serious Illness or "
                    "Injury of the Insured Person or an Immediate Family Member, or owing to the Insured "
                    "Person being summoned for jury service or as a witness, the Company shall reimburse "
                    "irrecoverable prepaid travel and accommodation costs up to USD 1,500.",
                    qa=[
                        QA(
                            "What is the trip cancellation limit under the Yatra Safe policy?",
                            "USD 1,500 for irrecoverable prepaid travel and accommodation costs.",
                            "numeric",
                            answer_keys=["1,500", "1500", "usd 1,500"],
                        )
                    ],
                    entities={"benefit": "Trip Cancellation", "limit_usd": 1500},
                ),
                Clause(
                    "3.2",
                    "Trip Delay",
                    "Where the scheduled departure of a common carrier is delayed by more than six (6) hours "
                    "owing to inclement weather, equipment failure or strike action, the Company shall pay "
                    "USD 50 for each full six-hour block of delay, up to a maximum of USD 300.",
                    qa=[
                        QA(
                            "How long must a flight be delayed before the Yatra Safe policy pays?",
                            "More than 6 hours, with USD 50 payable per full six-hour block up to USD 300.",
                            "numeric",
                            answer_keys=["6 hours", "six hours", "usd 50"],
                        )
                    ],
                    entities={"benefit": "Trip Delay", "trigger_hours": 6, "per_block_usd": 50, "cap_usd": 300},
                ),
                Clause(
                    "3.3",
                    "Checked-In Baggage Loss",
                    "In the event of permanent loss of checked-in baggage by a common carrier, the Company "
                    "shall pay up to USD 1,000, subject to a maximum of USD 500 per bag and USD 100 per "
                    "individual item. Any compensation received from the carrier shall be deducted from the "
                    "amount payable.",
                    qa=[
                        QA(
                            "What is the checked-in baggage loss limit under the Yatra Safe policy?",
                            "Up to USD 1,000 overall, with sub-limits of USD 500 per bag and USD 100 per item, less any carrier compensation.",
                            "numeric",
                            answer_keys=["1,000", "500", "100"],
                        )
                    ],
                    entities={"benefit": "Baggage Loss", "limit_usd": 1000, "per_bag_usd": 500},
                ),
                Clause(
                    "3.4",
                    "Baggage Delay",
                    "Where checked-in baggage is delayed by more than twelve (12) hours from the time of "
                    "arrival at the destination, the Company shall reimburse the cost of necessary emergency "
                    "purchases up to USD 200, supported by original receipts and written confirmation of the "
                    "delay from the carrier.",
                    qa=[
                        QA(
                            "How long must baggage be delayed to claim under the Yatra Safe policy?",
                            "More than 12 hours after arrival, reimbursing emergency purchases up to USD 200 against original receipts and carrier confirmation.",
                            "numeric",
                            answer_keys=["12 hours", "twelve", "usd 200"],
                        )
                    ],
                    entities={"benefit": "Baggage Delay", "trigger_hours": 12, "limit_usd": 200},
                ),
                Clause(
                    "3.5",
                    "Loss of Passport",
                    "The Company shall reimburse the actual expenses necessarily incurred in obtaining a "
                    "duplicate or fresh passport following its loss abroad, up to USD 300, provided the loss "
                    "was reported to the local police within twenty-four (24) hours of discovery.",
                    qa=[
                        QA(
                            "What does the Yatra Safe policy pay if I lose my passport abroad?",
                            "Up to USD 300 for obtaining a duplicate or fresh passport, provided the loss was reported to local police within 24 hours of discovery.",
                            "numeric",
                            answer_keys=["usd 300", "300", "24 hours"],
                        )
                    ],
                    entities={"benefit": "Loss of Passport", "limit_usd": 300, "police_hours": 24},
                ),
            ],
        ),
        Section(
            "4",
            "Personal Liability and Accident",
            [
                Clause(
                    "4.1",
                    "Personal Liability",
                    "The Company shall indemnify the Insured Person against legal liability to third parties "
                    "for accidental bodily injury or accidental damage to property occurring during the Trip, "
                    "up to USD 1,00,000. Liability arising out of the ownership or use of any motor vehicle, "
                    "aircraft or watercraft, or arising from any contractual undertaking, is excluded.",
                    qa=[
                        QA(
                            "What is the personal liability limit under the Yatra Safe policy?",
                            "USD 1,00,000, excluding liability from motor vehicles, aircraft, watercraft or contractual undertakings.",
                            "numeric",
                            answer_keys=["1,00,000", "100000", "100,000"],
                        )
                    ],
                    entities={"benefit": "Personal Liability", "limit_usd": 100000},
                ),
                Clause(
                    "4.2",
                    "Personal Accident",
                    "In the event of death or permanent total disablement of the Insured Person caused by an "
                    "Accident during the Trip, the Company shall pay the Sum Insured for this benefit as "
                    "stated in the Schedule. Permanent partial disablement shall be payable according to the "
                    "scale of benefits in Annexure C.",
                ),
            ],
        ),
        Section(
            "5",
            "Exclusions and Claims",
            [
                Clause(
                    "5.1",
                    "General Exclusions",
                    "No benefit shall be payable in respect of any claim arising from travel undertaken "
                    "against medical advice, travel for the purpose of obtaining medical treatment, "
                    "participation in professional sports, manual labour or hazardous occupation, pregnancy "
                    "or childbirth, mental or nervous disorders, or travel to a country for which the "
                    "Government of India has issued a travel advisory against travel.",
                    qa=[
                        QA(
                            "Does the Yatra Safe policy cover travel undertaken for medical treatment abroad?",
                            "No. Travel for the purpose of obtaining medical treatment is excluded.",
                            "boolean",
                            answer_keys=["no", "excluded"],
                        )
                    ],
                    entities={"exclusion": "General"},
                ),
                Clause(
                    "5.2",
                    "Claim Notification",
                    "All claims shall be notified to the Company's twenty-four hour assistance helpline as "
                    "soon as reasonably practicable and in any event within seven (7) days of the event. "
                    "Written claim documents shall be submitted within thirty (30) days of return to India.",
                    qa=[
                        QA(
                            "What is the claim notification deadline under the Yatra Safe travel policy?",
                            "Within 7 days of the event to the 24-hour helpline, with written documents within 30 days of return to India.",
                            "numeric",
                            answer_keys=["7 days", "30 days", "seven"],
                        )
                    ],
                    entities={"notify_days": 7, "doc_days_after_return": 30},
                ),
                Clause(
                    "5.3",
                    "Extension of Policy Period",
                    "The Policy Period may be extended once, for a maximum of one hundred and eighty (180) "
                    "additional days, provided the request is made before the expiry of the original Policy "
                    "Period, no claim has been made or is pending, and the additional premium is paid.",
                    qa=[
                        QA(
                            "Can I extend the Yatra Safe travel policy while abroad?",
                            "Yes, once, for up to 180 additional days, provided the request is made before expiry, no claim is made or pending, and additional premium is paid.",
                            "boolean",
                            answer_keys=["yes", "180", "once"],
                        )
                    ],
                    entities={"extension_days": 180, "extension_count": 1},
                ),
            ],
        ),
    ]
    return Policy(
        policy_id=pid,
        insurer="Sentinel Health Assurance Limited",
        product_name="Yatra Safe Overseas Travel Insurance",
        product_type="Overseas Travel",
        uin="SHALTRVL24022V012425",
        version="1.1",
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# Cross-policy (multi-hop) and unanswerable questions
# --------------------------------------------------------------------------------------

COMPARATIVE_QA: list[dict[str, Any]] = [
    {
        "question": "Which has the shorter pre-existing disease waiting period, Aarogya Suraksha or Nirogam Plus?",
        "answer": "Nirogam Plus, at 24 months of continuous coverage, compared with 36 months under Aarogya Suraksha.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#4.2", "HLT-NIROGAM#4.2"],
        "answer_keys": ["nirogam", "24", "36"],
    },
    {
        "question": "Compare the grace periods of the Aarogya Suraksha and Nirogam Plus health policies.",
        "answer": "Aarogya Suraksha allows 30 days. Nirogam Plus allows 15 days for annual premium and 30 days where premium is paid in instalments.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#8.2", "HLT-NIROGAM#7.4"],
        "answer_keys": ["30", "15"],
    },
    {
        "question": "Which health policy gives a better ambulance limit, Aarogya Suraksha or Nirogam Plus?",
        "answer": "Aarogya Suraksha, at Rs. 5,000 per hospitalisation event, against Rs. 3,000 per event under Nirogam Plus.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#3.4", "HLT-NIROGAM#3.5"],
        "answer_keys": ["aarogya", "5,000", "3,000"],
    },
    {
        "question": "How do the maternity waiting periods compare between Aarogya Suraksha and Nirogam Plus?",
        "answer": "Aarogya Suraksha requires 36 months of continuous coverage; Nirogam Plus requires 24 months and caps the benefit at Rs. 50,000 for normal delivery and Rs. 75,000 for caesarean.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#4.4", "HLT-NIROGAM#4.4"],
        "answer_keys": ["36", "24"],
    },
    {
        "question": "Compare the free look periods across the Aarogya Suraksha health policy and the Suraksha Kavach term plan.",
        "answer": "Aarogya Suraksha gives 15 days, extended to 30 days for distance marketing. Suraksha Kavach gives 30 days.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#8.1", "LIF-SURAKSHAKAVACH#5.4"],
        "answer_keys": ["15", "30"],
    },
    {
        "question": "I was hospitalised after a road accident and claimed under Aarogya Suraksha. Which waiting period applies?",
        "answer": "None. The 30-day initial waiting period expressly excludes claims arising out of an accident, and accident-related joint replacement is excluded from the specified disease waiting period.",
        "kind": "multi_hop",
        "gold_clauses": ["HLT-AAROGYA#4.1", "HLT-AAROGYA#4.3"],
        "answer_keys": ["no waiting", "none", "accident", "does not apply"],
    },
    {
        "question": "My car is four years old and I have zero depreciation cover under DriveShield. What depreciation applies to a replaced plastic bumper?",
        "answer": "None on the bumper. Zero depreciation removes depreciation on replaced parts except tyres, tubes and batteries, so the 50% plastic-parts rate does not apply.",
        "kind": "multi_hop",
        "gold_clauses": ["MOT-DRIVESHIELD#4.1", "MOT-DRIVESHIELD#5.1"],
        "answer_keys": ["nil", "no depreciation", "zero", "except tyres"],
    },
    {
        # Answerable, but only by contradicting the premise of the question — the
        # kind of case where a RAG system either cites the clause that refutes it or
        # invents a plausible table.
        "question": "What is the surrender value of the Suraksha Kavach term plan in year 10?",
        "answer": "There is none. The plan is a pure risk plan that acquires no surrender value, no paid-up value and no maturity value.",
        "kind": "contradicted_premise",
        "gold_clauses": ["LIF-SURAKSHAKAVACH#1.1"],
        "answer_keys": ["no surrender value"],
    },
    {
        "question": "Which of the two health policies restricts room rent by a percentage of sum insured rather than a room category?",
        "answer": "Aarogya Suraksha, which caps room rent at 1% of sum insured per day and 2% for ICU. Nirogam Plus instead entitles the insured to a single private air-conditioned room.",
        "kind": "comparative",
        "gold_clauses": ["HLT-AAROGYA#5.1", "HLT-NIROGAM#5.1"],
        "answer_keys": ["aarogya", "1%", "2%"],
    },
]


UNANSWERABLE_QA: list[dict[str, Any]] = [
    {
        "question": "What is the waiting period for dental implants under the DriveShield motor policy?",
        "answer": "Not covered by the indexed documents. The DriveShield policy is a motor policy and contains no dental provisions.",
        "kind": "unanswerable",
        "reason": "cross-domain confusion",
    },
    {
        "question": "What is the annual premium amount for the Aarogya Suraksha policy for a 40-year-old?",
        "answer": "Not stated in the indexed documents. Premium amounts appear in the policy schedule, not in the policy wording.",
        "kind": "unanswerable",
        "reason": "schedule-only data",
    },
    {
        "question": "Does the Nirogam Plus policy cover IVF and fertility treatment?",
        "answer": "Not addressed in the indexed documents. The Nirogam Plus wording contains no provision on infertility or assisted reproduction.",
        "kind": "unanswerable",
        "reason": "topic absent",
    },
    {
        "question": "What is the cashless hospital network size for the Yatra Safe travel policy in Europe?",
        "answer": "Not stated in the indexed documents. The Yatra Safe wording refers to an assistance helpline but gives no network size.",
        "kind": "unanswerable",
        "reason": "topic absent",
    },
    {
        "question": "What is the no claim bonus under the Griha Raksha householder's policy?",
        "answer": "Not stated in the indexed documents. The Griha Raksha wording contains no no-claim-bonus provision.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "How much does Meridian General charge for the zero depreciation add-on?",
        "answer": "Not stated in the indexed documents. The wording describes the add-on's scope but not its price.",
        "kind": "unanswerable",
        "reason": "pricing absent",
    },
    {
        "question": "What is the claim settlement ratio of Sentinel Health Assurance Limited?",
        "answer": "Not stated in the indexed documents. Claim settlement ratios are regulatory disclosures, not policy-wording content.",
        "kind": "unanswerable",
        "reason": "out of corpus",
    },
    {
        "question": "Does the Aarogya Suraksha policy cover gender reassignment surgery?",
        "answer": "Not addressed in the indexed documents. The Aarogya Suraksha wording contains no provision on this treatment.",
        "kind": "unanswerable",
        "reason": "topic absent",
    },
    {
        "question": "What is the sum insured on my Aarogya Suraksha policy?",
        "answer": "Not stated in the indexed documents. The sum insured is an individual figure recorded in the policy schedule, not in the wording.",
        "kind": "unanswerable",
        "reason": "schedule-only data",
    },
    {
        "question": "Which hospitals near Hyderabad are in the Nirogam Plus cashless network?",
        "answer": "Not stated in the indexed documents. The wording refers to a network list published separately by the insurer.",
        "kind": "unanswerable",
        "reason": "external list",
    },
    {
        "question": "What is the grace period under the Yatra Safe overseas travel policy?",
        "answer": "Not stated in the indexed documents. The Yatra Safe wording contains no grace period provision.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "What cumulative bonus does the Nirogam Plus policy give for a claim-free year?",
        "answer": "Not stated in the indexed documents. Nirogam Plus offers a premium discount rather than a cumulative bonus, and no cumulative bonus clause exists in its wording.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "What is the maternity benefit under the DriveShield motor policy?",
        "answer": "Not covered by the indexed documents. DriveShield is a motor policy and has no maternity provision.",
        "kind": "unanswerable",
        "reason": "cross-domain confusion",
    },
    {
        "question": "How many riders can I attach to the Griha Raksha householder's policy?",
        "answer": "Not stated in the indexed documents. The Griha Raksha wording describes sections rather than riders and sets no rider limit.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "What is the IRDAI registration number of Meridian General Insurance Company Limited?",
        "answer": "Not stated in the indexed documents. The wordings give product UINs, not the insurer's registration number.",
        "kind": "unanswerable",
        "reason": "out of corpus",
    },
    {
        "question": "What GST rate applies to the premium on the Suraksha Kavach term plan?",
        "answer": "Not stated in the indexed documents. Tax treatment is not addressed in the policy wording.",
        "kind": "unanswerable",
        "reason": "out of corpus",
    },
    {
        "question": "Does the Yatra Safe travel policy cover adventure sports like bungee jumping?",
        "answer": "Not resolved by the indexed documents. The Yatra Safe wording excludes professional sports but says nothing about recreational adventure activities such as this.",
        "kind": "unanswerable",
        "reason": "partially related exclusion only",
    },
    {
        "question": "What is the deductible on the Griha Raksha burglary section?",
        "answer": "Not stated in the indexed documents. The Griha Raksha wording sets no deductible for burglary cover.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "How long does Meridian General take to issue a renewal notice before expiry?",
        "answer": "Not stated in the indexed documents. No renewal-notice timeline appears in these wordings.",
        "kind": "unanswerable",
        "reason": "topic absent",
    },
    {
        "question": "What is the free look period under the DriveShield motor policy?",
        "answer": "Not stated in the indexed documents. The DriveShield wording contains no free look provision.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
    {
        "question": "What portability credit does the Yatra Safe travel policy give?",
        "answer": "Not stated in the indexed documents. Portability is a health-policy provision and does not appear in the Yatra Safe wording.",
        "kind": "unanswerable",
        "reason": "concept exists in another policy",
    },
]


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

BUILDERS = [
    build_health_aarogya,
    build_health_nirogam,
    build_motor_package,
    build_term_life,
    build_home_package,
    build_travel,
]


def build_all() -> list[Policy]:
    return [b() for b in BUILDERS]


def build_manifest(policies: list[Policy]) -> dict[str, Any]:
    out: dict[str, Any] = {"policies": [], "clauses": []}
    for p in policies:
        out["policies"].append(
            {
                "policy_id": p.policy_id,
                "insurer": p.insurer,
                "product_name": p.product_name,
                "product_type": p.product_type,
                "uin": p.uin,
                "version": p.version,
                "filename": f"{p.policy_id}.txt",
                "display": p.display,
            }
        )
        for s in p.sections:
            for c in s.clauses:
                out["clauses"].append(
                    {
                        "clause_id": _clause_id(p.policy_id, c.number),
                        "policy_id": p.policy_id,
                        "section_number": s.number,
                        "section_title": s.title,
                        "clause_number": c.number,
                        "clause_title": c.title,
                        "entities": c.entities,
                        "n_chars": len(c.text),
                    }
                )
    return out


def build_golden_qa(policies: list[Policy]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qid = 0
    for p in policies:
        for s in p.sections:
            for c in s.clauses:
                for qa in c.qa:
                    qid += 1
                    rows.append(
                        {
                            "id": f"Q{qid:03d}",
                            "question": qa.question,
                            "reference_answer": qa.answer,
                            "kind": qa.kind,
                            "answerable": True,
                            "gold_clauses": [_clause_id(p.policy_id, c.number)],
                            "gold_policies": [p.policy_id],
                            "answer_keys": qa.answer_keys or [],
                        }
                    )
    for row in COMPARATIVE_QA:
        qid += 1
        policies_touched = sorted({c.split("#")[0] for c in row["gold_clauses"]})
        rows.append(
            {
                "id": f"Q{qid:03d}",
                "question": row["question"],
                "reference_answer": row["answer"],
                "kind": row["kind"],
                "answerable": True,
                "gold_clauses": row["gold_clauses"],
                "gold_policies": policies_touched,
                "answer_keys": row.get("answer_keys", []),
            }
        )
    for row in UNANSWERABLE_QA:
        qid += 1
        rows.append(
            {
                "id": f"Q{qid:03d}",
                "question": row["question"],
                "reference_answer": row["answer"],
                "kind": row["kind"],
                "answerable": False,
                "gold_clauses": row.get("gold_clauses", []),
                "gold_policies": sorted({c.split("#")[0] for c in row.get("gold_clauses", [])}),
                "answer_keys": [],
                "unanswerable_reason": row["reason"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data", help="output root (default: data)")
    args = ap.parse_args()

    root = Path(args.out_dir)
    corpus_dir = root / "corpus"
    eval_dir = root / "eval"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    policies = build_all()
    for p in policies:
        (corpus_dir / f"{p.policy_id}.txt").write_text(render_policy(p), encoding="utf-8")

    manifest = build_manifest(policies)
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa = build_golden_qa(policies)
    (eval_dir / "golden_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    n_answerable = sum(1 for q in qa if q["answerable"])
    n_multi = sum(1 for q in qa if len(q["gold_clauses"]) > 1)
    print(f"policies     : {len(policies)}")
    print(f"clauses      : {len(manifest['clauses'])}")
    print(f"questions    : {len(qa)}  (answerable {n_answerable}, unanswerable {len(qa) - n_answerable})")
    print(f"multi-clause : {n_multi}")
    print(f"written to   : {corpus_dir} and {eval_dir}")


if __name__ == "__main__":
    main()
