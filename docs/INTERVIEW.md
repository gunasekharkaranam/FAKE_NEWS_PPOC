# Interview notes

## 30-second pitch

> "I built a fake news checker. My first version used TF-IDF and logistic regression on writing style. When I tested it properly, I found two problems: it dropped to about 0.5 macro-F1 on datasets it hadn't seen, and on calm, well-written false sentences it was 52% accurate, basically a coin flip. So I redesigned it as a fact-checker. It extracts the claims, searches Google's fact-check database and Wikipedia for evidence, and uses an NLI model trained on the FEVER fact-verification dataset to decide whether the evidence supports or contradicts each claim. Evidence decides; style is only a fallback hint, clearly labelled. It's a FastAPI app with tests, CI and Docker."

## "Can you do it on real-world data?"

> "Yes. I moved from a Kaggle dataset to real fact-checked data: 12.8k PolitiFact claims from LIAR and about 21k headlines from FakeNewsNet. I also ran a cross-dataset test: train on one source, test on another. The style model drops to near random, which shows style doesn't generalise. That's why the final decision is made by evidence from live sources, which works on any new text, not only text that looks like the training data."

## Likely questions

**Why macro-F1 and not accuracy?**
GossipCop is 76% real. A model that always says "real" gets 76% accuracy but is useless. Macro-F1 averages the F1 of both classes.

**Why is LIAR so hard (~0.56–0.61)?**
The claims are one short, calm sentence. There are few style clues, and truth depends on facts the text doesn't contain. Published text-only results on binary LIAR are also low.

**How do you explain predictions?**
For logistic regression, word contribution = tf-idf value × weight. This is exact. For the fact-check part, the explanation is the evidence itself: the source link and sentence.

**What is NLI?**
Natural Language Inference. Given two texts, it decides if the first supports, contradicts or is neutral to the second. I use the evidence as the premise and the claim as the hypothesis.

**What if the evidence is about a different event?**
For fact-checker ratings I require at least 30% keyword overlap before using the rating. For NLI, the threshold is 0.6, and the user always sees the source sentence.

**What if there is no evidence (breaking news)?**
The app says "Unverified" and shows the style guess, labelled as a guess. Being honest beats being confident and wrong.

**Topic bias?**
The style model links "obama" and "trump" with fake, because those are what fact-checkers chose to check. That's dataset bias, and one more reason style is not the final judge.

**How would you scale it?**
Cache evidence per claim, run NLI in batches on a GPU worker, add a news API for fresh events, and retrain from the feedback table.

**Security?**
SSRF protection on URL fetching (blocks private IPs, limits redirects, size and time), XSS-safe rendering, input length limits, and the Docker container runs as a non-root user.

**What would you do next?**
Add a news search API for recent events, evaluate on FEVER, and use an LLM to rewrite messy text into clean claims before verification.
