"""Hand-written, app-themed content for the GapSound workload.

Four apps (organizations) with system prompts of deliberately different lengths, as real products have.
User messages are templates with slots so every user's text is unique (identical messages across users are
rare in real traffic and would create cache sharing the experiment did not intend).
Message lengths follow a short/medium/long-paste mixture chosen to match the mean user-prompt length reported
for LMSYS-Chat-1M (~70 tokens; Zheng et al., ICLR 2024).
"""

SYSTEM = {
"tax": (
"You are TaxHelper, the virtual assistant for Northfield Tax Services. Answer general questions about US federal "
"and state income tax filing, deductions, credits, deadlines, and our office's services. Be concise. Do not give "
"advice that depends on facts you do not have; ask a clarifying question instead. Remind users that a licensed "
"preparer must review their return before filing. Office hours: Monday to Saturday, 9am to 6pm. Never ask for a "
"full Social Security number."
),
"hr": (
"You are the Summit Manufacturing HR Benefits Assistant. You help employees understand company benefits and "
"policies. Topics you cover: medical, dental and vision plans (Blue Ridge PPO, Blue Ridge HDHP with HSA, "
"ClearSight vision, BrightSmile dental); the 401(k) plan with a 4% company match vesting over three years; paid "
"time off (15 days in years one to four, 20 days from year five, plus 10 company holidays); sick leave; parental "
"leave (12 weeks paid for birthing parents, 6 weeks for other parents); FMLA; short-term and long-term disability; "
"the employee assistance program; tuition reimbursement up to $5,250 per year; and timesheet and payroll questions.\n"
"Rules: answer from the policies above and say when something must be confirmed with HR. Do not discuss other "
"employees' information. For complaints about harassment or discrimination, direct the employee to the confidential "
"HR line at extension 4400 or the ethics hotline. Open enrollment runs November 1 to November 15; outside it, plan "
"changes require a qualifying life event reported within 30 days."
),
"clinic": (
"You are the patient services assistant for Riverside Family Clinic, a primary care practice with locations in "
"Riverside and Oak Park. You help patients with scheduling, prescriptions, billing, insurance, test results "
"logistics, and general health information.\n\n"
"What you can do:\n"
"- Explain how to book, reschedule, or cancel appointments (online portal, phone 555-0142, or front desk). Same-day "
"sick visits open at 7:30am.\n"
"- Explain prescription refill requests (allow 2 business days; controlled substances require a visit).\n"
"- Explain billing statements, copays, deductibles, and coinsurance in general terms, and how to set up a payment "
"plan (balances over $200, up to 12 months, no interest).\n"
"- List accepted insurance: Blue Cross PPO and HMO, Aetna, UnitedHealthcare, Cigna, Medicare, and state Medicaid.\n"
"- Explain how lab and imaging results are released: normal results post to the portal within 3 business days; the "
"care team calls about abnormal results.\n"
"- Give general health information from reputable sources, clearly marked as general.\n\n"
"What you must not do:\n"
"- Do not diagnose, interpret individual test results, or recommend changing a medication. Say that the patient's "
"clinician will review it.\n"
"- If the user describes chest pain, trouble breathing, stroke symptoms, severe bleeding, or thoughts of self-harm, "
"tell them to call 911 or go to the nearest emergency department immediately.\n"
"- Do not ask for or repeat full insurance member IDs, Social Security numbers, or payment card numbers.\n\n"
"Tone: warm, plain language, short paragraphs. Patients may be anxious; acknowledge concerns before giving "
"logistics. Always offer the next concrete step."
),
"legal": (
"You are the intake and information assistant for Harbor Legal Aid, a nonprofit that provides free civil legal "
"help to low-income residents of Bay County. You are not a lawyer and you do not give legal advice. You provide "
"general legal information, help people understand their options and deadlines, screen for eligibility, and help "
"them prepare for a consultation with a staff attorney.\n\n"
"Practice areas we cover:\n"
"1. Housing: evictions, lease disputes, security deposits, repairs and habitability, lockouts and utility "
"shutoffs, and housing discrimination.\n"
"2. Consumer: debt collection lawsuits, wage garnishment, repossession, and predatory lending.\n"
"3. Family: protective orders, custody and support modifications, and uncontested divorce for survivors of "
"domestic violence.\n"
"4. Public benefits: denials or terminations of SNAP, Medicaid, SSI and SSDI, and unemployment insurance.\n"
"5. Employment: unpaid wages and wrongful termination intake (we refer most employment cases to partner "
"organizations).\n\n"
"We do not handle criminal cases, immigration cases, personal injury, or business disputes. For those, refer the "
"person to the county bar association lawyer referral service at 555-0199.\n\n"
"Eligibility: household income at or below 200% of the federal poverty guidelines, residence in Bay County, and "
"a civil legal problem in one of our practice areas. Seniors aged 60 and over and survivors of domestic violence "
"may qualify regardless of income.\n\n"
"How to respond:\n"
"- Start by understanding the problem. Ask one or two focused questions if key facts are missing, especially "
"dates, notices received, and any court dates.\n"
"- Deadlines matter. If the person mentions a court date, a summons, or a notice with a deadline, tell them the "
"deadline is important, that missing it can mean losing the case by default, and that they should contact our "
"intake line at 555-0177 the same day.\n"
"- In Bay County, a tenant served with an eviction summons generally has 5 business days to file a written answer "
"with the court. A landlord must return a security deposit or give an itemized list of deductions within 30 days "
"after the tenant moves out. Present these as general information, not advice about the person's case.\n"
"- Never tell a person they will win or lose. Never draft court filings. You may explain what a document is and "
"what information an attorney will need.\n"
"- If the person describes immediate danger or domestic violence, give the 24-hour hotline 555-0100 and 911 first.\n"
"- Do not ask for Social Security numbers, immigration status, or bank account numbers.\n\n"
"Style: plain language at about an eighth-grade reading level, short sentences, and a calm, respectful tone. "
"Many people contacting us are under stress. End each answer with a clear next step."
),
}

# ---------------------------------------------------------------- slot values
NAMES = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Tom", "Fatima", "Daniel", "Keisha", "Olga", "Luis",
         "Grace", "Omar", "Hannah", "Diego", "Mei", "Samuel", "Rosa", "Kevin", "Nadia", "Ethan", "Leila", "Marcus"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

# Each bank: list of (length_class, template). length_class in {"short", "medium", "long"}.
# {n}: a number slot, {name}: a first name, {month}: a month, {amt}: a dollar amount, {d}: a day 1-28.
OPENERS = {
"tax": [
 ("short", "When is the deadline to file my taxes this year?"),
 ("short", "Can I still file if I missed the April deadline by {n} days?"),
 ("short", "How much do you charge to prepare a simple return?"),
 ("short", "Do I need to file taxes if I only made ${amt} last year?"),
 ("medium", "I started freelancing in {month} as a graphic designer and made about ${amt}. I didn't pay any "
            "estimated taxes. Am I going to get a penalty, and what records should I bring to my appointment?"),
 ("medium", "My {n}-year-old son lives with me most of the year but his dad wants to claim him. We were never married. "
            "Who gets to claim him and the child tax credit?"),
 ("medium", "I got a letter from the IRS saying I owe ${amt} from a return {n} years ago. I don't remember any "
            "problem with that return. What should I do first?"),
 ("medium", "I work from home {n} days a week for my employer. Can I deduct part of my rent and internet bill?"),
 ("long", "Here is what my W-2 says, can you tell me if anything looks off before I file?\n"
          "Box 1 Wages: ${amt}\nBox 2 Federal income tax withheld: ${amt2}\nBox 3 Social security wages: ${amt}\n"
          "Box 4 Social security tax withheld: ${amt3}\nBox 5 Medicare wages: ${amt}\nBox 6 Medicare tax withheld: "
          "${amt4}\nBox 12a Code D: ${amt5}\nBox 15 State: MI\nBox 16 State wages: ${amt}\nBox 17 State income tax: "
          "${amt6}\nI also had a second job from {month} to December that paid about ${amt7}, but I haven't received that "
          "W-2 yet. My employer said it was mailed on the {d}th. Should I wait for it or file now and amend later? I "
          "would rather not pay to amend if I can avoid it."),
],
"hr": [
 ("short", "How many PTO days do I have left this year?"),
 ("short", "When does open enrollment end?"),
 ("short", "Is the dental plan changing next year?"),
 ("short", "Who do I talk to about a payroll error on my last check?"),
 ("medium", "I'm having a baby in {month}. How much paid parental leave do I get and do I need to apply for FMLA "
            "separately? I've been with the company for {n} years."),
 ("medium", "I'm trying to decide between the PPO and the HDHP. I'm healthy and only see a doctor about {n} times a "
            "year, but my spouse has a prescription that costs about ${amt} a month. Which one usually makes sense?"),
 ("medium", "My manager told me I can't take vacation in {month} because of the busy season. Is that allowed under "
            "the policy? I requested the days {n} weeks ago."),
 ("medium", "I want to go back to school for an accounting degree. How does tuition reimbursement work and do I have "
            "to stay with the company for a certain time after?"),
 ("long", "My last paycheck looks wrong. Here are the lines from my pay stub:\nRegular hours: 80.00 at $24.50 = "
          "$1,960.00\nOvertime hours: {n}.00 at $36.75\nHoliday pay: 8.00 at $24.50 = $196.00\nGross: ${amt}\n"
          "Federal withholding: ${amt2}\nState withholding: ${amt3}\nSocial Security: ${amt4}\nMedicare: ${amt5}\n"
          "401(k) 6%: ${amt6}\nMedical PPO: $142.30\nDental: $18.20\nNet pay: ${amt7}\nI worked the {d}th and the "
          "holiday, and I think my overtime from the week of {month} {d} is missing completely. My timesheet was "
          "approved by my supervisor {name}. What should I do and how long does a correction usually take?"),
],
"clinic": [
 ("short", "How do I request a prescription refill?"),
 ("short", "Can I get a same-day appointment today?"),
 ("short", "Do you take Aetna insurance?"),
 ("short", "When will my blood test results be ready?"),
 ("medium", "I got a bill for ${amt} for my visit on {month} {d}, but I thought my insurance covered annual "
            "physicals at 100%. Why was I charged and how do I dispute it?"),
 ("medium", "My {n}-year-old daughter has had a fever of 101 for two days and a sore throat. Should I bring her in "
            "or wait? What times are open tomorrow at the Oak Park location?"),
 ("medium", "I'm switching jobs in {month} and my new insurance is UnitedHealthcare. Do I need to do anything before "
            "my next appointment with Dr. {name}?"),
 ("medium", "I've been feeling really tired for about {n} weeks and my sleep has been bad. I don't know if I need an "
            "appointment or if this is normal stress. What do you suggest?"),
 ("long", "I just saw these results in the portal and I'm worried. Can you explain what they mean?\n"
          "COMPREHENSIVE METABOLIC PANEL, collected {month} {d}\nGlucose 1{n} mg/dL (ref 70-99) H\nBUN 18 mg/dL "
          "(ref 7-20)\nCreatinine 1.1 mg/dL (ref 0.6-1.2)\nSodium 139 mmol/L (ref 135-145)\nPotassium 4.6 mmol/L "
          "(ref 3.5-5.1)\nChloride 101 mmol/L (ref 98-107)\nCO2 25 mmol/L (ref 22-29)\nCalcium 9.4 mg/dL "
          "(ref 8.6-10.3)\nTotal protein 7.1 g/dL (ref 6.0-8.3)\nAlbumin 4.3 g/dL (ref 3.5-5.0)\nALT 4{n} U/L "
          "(ref 7-56)\nAST 3{n} U/L (ref 10-40)\nHEMOGLOBIN A1c 6.{n}% (ref <5.7) H\nNobody has called me yet. My "
          "mother had diabetes and I am scared this means I have it too. I am {n}0 years old and I take lisinopril "
          "for blood pressure. Should I stop eating sugar right away or wait for Dr. {name} to call?"),
],
"legal": [
 ("short", "Can my landlord change the locks without a court order?"),
 ("short", "Do you help with child custody cases?"),
 ("short", "How do I know if I qualify for free help?"),
 ("short", "I got court papers about a debt. What do I do?"),
 ("medium", "My landlord kept my whole ${amt} security deposit. I moved out on {month} {d} and left the apartment "
            "clean. It has been {n} weeks and I haven't gotten anything in writing. What are my options?"),
 ("medium", "I was cut off from SNAP benefits last month because they said I didn't turn in a form, but I mailed it "
            "on {month} {d}. I have {n} kids. How do I appeal and how long do I have?"),
 ("medium", "A debt collector called saying I owe ${amt} on a credit card from {n} years ago. I don't recognize the "
            "company. Do I have to pay this and can they sue me?"),
 ("medium", "My employer hasn't paid me for my last {n} weeks of work and now they won't answer my calls. I worked "
            "as a cook. Can you help me get my wages?"),
 ("long", "I got this paper taped to my door yesterday. Here is what it says:\n\"NOTICE TO QUIT FOR NONPAYMENT OF "
          "RENT. To: {name} and all occupants. You are hereby notified that you are in default in payment of rent "
          "for the premises in the amount of ${amt}, representing rent due for the months of {month} and the "
          "following month. Demand is hereby made that you pay said rent in full or deliver possession of the premises "
          "within SEVEN (7) days after service of this notice. If you fail to do so, legal proceedings will be "
          "instituted against you to recover possession of the premises, unpaid rent, court costs, and attorney "
          "fees as permitted by law and the lease.\" It is signed by the property manager. I lost my job in "
          "{month} and I have been paying what I can, about ${amt2} a month. I have {n} children and nowhere else "
          "to go. Does this mean I have to be out in 7 days? Can they put my things on the street?"),
],
}

FOLLOWUPS = {
"tax": [
 ("short", "What if I can't pay all of it at once?"),
 ("short", "Can you book me an appointment for next week?"),
 ("short", "Thanks. What documents should I bring?"),
 ("short", "Is that the same for state taxes?"),
 ("medium", "OK, and if I set up a payment plan with the IRS, does interest keep adding up the whole time? I could "
            "probably pay about ${amt} a month."),
 ("medium", "My employer also gave me a ${amt} bonus in December. Does that change which tax bracket I'm in?"),
 ("medium", "I forgot to mention I also sold some stock in {month} for a loss of about ${amt}. Does that help me?"),
 ("long", "Here is the IRS letter text, maybe this helps: \"Notice CP2000. We are proposing changes to your Form "
          "1040 for tax year {n}. The income and payment information we have on file from your employers, banks, and "
          "other payers does not match the information you reported. Proposed amount due: ${amt}. This includes "
          "tax, penalty, and interest. If you agree, sign the response form and pay by {month} {d}. If you do not "
          "agree, send a signed statement and supporting documents explaining why.\" The income they list is from "
          "a 1099-K for ${amt2} from selling my old furniture online, which I sold for less than I paid."),
],
"hr": [
 ("short", "Where do I submit that form?"),
 ("short", "Does that count against my PTO?"),
 ("short", "Can my spouse be added mid-year?"),
 ("short", "Who is my HR business partner?"),
 ("medium", "If I switch to the HDHP, how much does the company put into the HSA, and can I use it for my spouse's "
            "prescription?"),
 ("medium", "My doctor says I need surgery in {month} and about {n} weeks of recovery. Do I use short-term disability "
            "or FMLA or both?"),
 ("medium", "I've been here {n} years. If I leave in {month}, how much of the 401(k) match do I get to keep?"),
 ("long", "Here's the email my manager sent me, I want to know if this follows policy: \"Hi {name}, after reviewing "
          "the schedule for Q{n}, I can't approve any PTO between {month} 1 and the end of the following month. "
          "Requests already approved during that window will need to be rescheduled. I know this is frustrating, "
          "but we're short-staffed and the line needs full coverage. If you have a family emergency, come talk to "
          "me directly and we'll figure something out. Thanks for understanding.\" I already had {n} days approved "
          "for my sister's wedding and the flights are nonrefundable, about ${amt}."),
],
"clinic": [
 ("short", "OK, can I book that online?"),
 ("short", "Is there a cost for the portal?"),
 ("short", "What should I bring to the appointment?"),
 ("short", "How long is the usual wait?"),
 ("medium", "Thanks. If I set up a payment plan for the ${amt}, does it affect whether Dr. {name} will still see me?"),
 ("medium", "She also has a rash on her arms now, small red dots. Does that change anything? She's had all her "
            "vaccines."),
 ("medium", "The pharmacy said the refill needs prior authorization from my insurance. Who handles that, you or them?"),
 ("long", "Here's the explanation of benefits my insurance sent: \"Date of service {month} {d}. Provider Riverside "
          "Family Clinic. Service: Office visit, established patient, moderate complexity (99214). Billed ${amt}. "
          "Allowed amount ${amt2}. Plan paid $0.00. Applied to deductible ${amt2}. Patient responsibility ${amt2}. "
          "Remark code N130: consult plan benefit documents for coverage limitations.\" But I came in for my annual "
          "physical and only mentioned my knee hurt at the end. Why was it billed as a regular visit?"),
],
"legal": [
 ("short", "What should I bring to the consultation?"),
 ("short", "Can I do this without a lawyer?"),
 ("short", "Do I have to go to court in person?"),
 ("short", "What is the intake line number again?"),
 ("medium", "The court date on the summons is {month} {d}. Is it too late to file an answer? I received it {n} days "
            "ago."),
 ("medium", "My income is about ${amt} a month and I live with my {n} kids. Do I qualify for free help?"),
 ("medium", "If I pay part of what they say I owe, does that restart the clock on the old debt?"),
 ("long", "Here is the summons text: \"STATE OF MICHIGAN, BAY COUNTY DISTRICT COURT. SUMMONS AND COMPLAINT, LANDLORD-"
          "TENANT. Plaintiff: Harborview Properties LLC. Defendant: {name}. The plaintiff seeks possession of the "
          "premises and a money judgment of ${amt}. You must appear at the hearing on {month} {d} at 9:00 a.m. "
          "If you do not appear, a default judgment may be entered against you for possession and the amount "
          "claimed. You may file a written answer before the hearing.\" They also added late fees of ${amt2} that "
          "I never agreed to. What does default judgment mean?"),
],
}

# Words used to generate per-probe "pasted documents" in each app's theme (other-user probe).
DOC_WORDS = {
"tax": "return deduction refund income filing receipt audit credit dependent withholding schedule quarterly estimate "
       "penalty bracket basis depreciation contribution adjusted gross exemption amended".split(),
"hr": "benefits enrollment leave payroll retirement dental vision policy manager overtime timesheet onboarding bonus "
      "vacation handbook accrual eligibility premium beneficiary vesting".split(),
"clinic": "patient appointment referral laboratory prescription copay insurance physician nurse visit record consent "
          "pharmacy billing diagnosis follow-up imaging specimen dosage allergy".split(),
"legal": "tenant landlord lease notice hearing summons complaint judgment deposit appeal deadline evidence settlement "
         "garnishment defendant plaintiff possession affidavit motion statute".split(),
}
