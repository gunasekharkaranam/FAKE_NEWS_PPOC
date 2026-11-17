// Front-end logic. All server text is inserted with textContent (never innerHTML)
// so a malicious article cannot inject HTML/JS into the page.
const $ = (id) => document.getElementById(id);
let mode = "text";
let last = null;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.tab;
    $("text").hidden = mode !== "text";
    $("url").hidden = mode !== "url";
  })
);

document.querySelectorAll(".ex").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelector('[data-tab="text"]').click();
    $("text").value = b.textContent;
  })
);

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { use_evidence: $("evidence").checked };
  if (mode === "url") body.url = $("url").value.trim();
  else body.text = $("text").value.trim();
  if (!body.url && !body.text) return;

  $("error").hidden = true;
  $("result").hidden = true;
  $("loading").hidden = false;
  $("go").disabled = true;
  try {
    const r = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Something went wrong.");
    last = { text: body.text || body.url, verdict: data.verdict };
    render(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    $("loading").hidden = true;
    $("go").disabled = false;
  }
});

function render(d) {
  const kind = d.verdict.startsWith("Likely fake") ? "fake"
             : d.verdict.startsWith("Likely real") ? "real" : "unsure";
  $("verdict").className = "card verdict " + kind;
  $("verdict-label").textContent = d.verdict;
  $("verdict-reason").textContent = d.reason;
  const by = { evidence: "Decided by evidence", style: "Decided by writing style only", none: "Not decided" };
  let decided = by[d.decided_by] || "";
  if (!d.nli_enabled) decided += " · NLI model off (fact-checker ratings only)";
  if (d.source_errors.length) decided += " · some sources failed: " + d.source_errors.join(", ");
  if (d.article) decided += " · article: " + d.article.title;
  $("decided").textContent = decided;

  // claims + evidence
  const box = $("claims");
  box.replaceChildren();
  if (!d.claims.length) box.append(el("p", "muted", "Fact checking was turned off."));
  d.claims.forEach((c) => {
    const item = el("div", "claim");
    const head = el("div", "claim-head");
    head.append(el("span", "badge " + c.verdict.split(" ")[0].toLowerCase(), c.verdict));
    head.append(el("span", "claim-text", "“" + c.claim + "”"));
    item.append(head);
    if (!c.evidence.length) item.append(el("p", "muted", "No evidence found."));
    c.evidence.forEach((ev) => {
      const row = el("div", "ev");
      const a = el("a", "", (ev.publisher || ev.source) + ": " + ev.title);
      a.href = ev.url; a.target = "_blank"; a.rel = "noopener noreferrer";
      row.append(a);
      const info = ev.rating ? `Rating: ${ev.rating} · ` : "";
      row.append(el("div", "ev-text", ev.text));
      row.append(el("div", "ev-meta", `${info}${ev.verdict} (${ev.method}, ${Math.round(ev.confidence * 100)}%)`));
      item.append(row);
    });
    box.append(item);
  });

  // style
  const s = d.style;
  const pct = Math.round(s.fake_probability * 100);
  $("meter-fill").style.width = pct + "%";
  $("style-num").textContent = s.label === "Out of scope" ? "unknown words" : pct + "% fake-style";
  const words = $("words");
  words.replaceChildren();
  s.top_words.forEach((w) =>
    words.append(el("span", "chip " + (w.weight > 0 ? "neg" : "pos"), (w.weight > 0 ? "▲ " : "▼ ") + w.word))
  );
  const f = s.features;
  $("signals").replaceChildren(
    el("li", "", `Exclamation marks: ${f.exclamations}`),
    el("li", "", `ALL-CAPS words: ${f.all_caps_words}`),
    el("li", "", `Words: ${f.word_count}`)
  );
  $("thanks").hidden = true;
  $("result").hidden = false;
}

document.querySelectorAll(".feedback button").forEach((b) =>
  b.addEventListener("click", async () => {
    if (!last) return;
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...last, correct: b.dataset.ok === "1" }),
    });
    $("thanks").hidden = false;
  })
);
