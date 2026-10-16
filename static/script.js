"use strict";

// Same origin when served by FastAPI (Render). Falls back to local server if the file is opened directly.
const API_BASE = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const $ = (id) => document.getElementById(id);
const form = $("loanForm");
const btn = $("submitBtn");
const errorBox = $("errorBox");
const gauge = document.querySelector(".gauge");
const card = $("resultCard");

gauge.classList.add("idle");

/* ---------- live helpers ---------- */
function updateLive() {
  $("rateOut").textContent = Number(form.loan_int_rate.value).toFixed(1) + "%";

  const income = parseFloat(form.person_income.value);
  const loan = parseFloat(form.loan_amnt.value);
  const ratio = income > 0 && loan >= 0 ? loan / income : 0;
  $("ratioOut").textContent = Math.round(ratio * 100) + "%";
  const bar = $("ratioBar");
  bar.style.width = Math.min(ratio * 100 / 0.8, 100) + "%"; // full bar at 80% of income
  bar.style.background = ratio > 0.4 ? "var(--red)" : ratio > 0.25 ? "var(--amber)" : "var(--teal)";
  return ratio;
}
form.addEventListener("input", updateLive);
updateLive();

/* ---------- example data ---------- */
const SAMPLES = {
  safe: {
    person_age: 38, person_income: 95000, person_home_ownership: "MORTGAGE", person_emp_length: 10,
    loan_intent: "HOMEIMPROVEMENT", loan_amnt: 8000, loan_int_rate: 7.5, loan_grade: "A",
    cb_person_default_on_file: "N", cb_person_cred_hist_length: 14,
  },
  risky: {
    person_age: 23, person_income: 22000, person_home_ownership: "RENT", person_emp_length: 1,
    loan_intent: "MEDICAL", loan_amnt: 15000, loan_int_rate: 20.5, loan_grade: "F",
    cb_person_default_on_file: "Y", cb_person_cred_hist_length: 2,
  },
};

document.querySelectorAll("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => {
    const s = SAMPLES[b.dataset.sample];
    for (const [key, val] of Object.entries(s)) {
      const els = form.elements[key];
      if (els instanceof RadioNodeList) els.value = val;
      else els.value = val;
    }
    updateLive();
    form.requestSubmit();
  })
);

/* ---------- gauge ---------- */
function setGauge(p, threshold) {
  gauge.classList.remove("idle");
  const angle = -90 + p * 180;
  $("gFill").style.strokeDashoffset = 377 * (1 - p);
  $("gNeedle").style.transform = `rotate(${angle}deg)`;
  const t = $("gThresh");
  t.setAttribute("transform", `rotate(${-90 + threshold * 180} 150 150)`);
  t.style.opacity = 1;
}

function countUp(el, target, ms = 1400) {
  const start = performance.now();
  const from = parseFloat(el.textContent) || 0;
  (function tick(now) {
    const k = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = (from + (target - from) * eased).toFixed(1);
    if (k < 1) requestAnimationFrame(tick);
  })(start);
}

function showResult(r) {
  const p = r.default_probability;
  const high = r.default_prediction === 1;

  setGauge(p, r.threshold);
  countUp($("pctNum"), r.default_probability_percentage);

  card.classList.remove("low", "high");
  card.classList.add(high ? "high" : "low");

  $("verdict").textContent = r.decision;
  const badge = $("badge");
  badge.hidden = false;
  badge.textContent = r.risk;
  badge.className = "badge " + (high ? "high" : "low");
  void badge.offsetWidth; // restart animation
  badge.classList.add("pop");

  $("thrOut").textContent = (r.threshold * 100).toFixed(1) + "%";
  $("classOut").textContent = high ? "Default (1)" : "Repay (0)";
  $("facts").hidden = false;
  $("hint").textContent = high
    ? "The default probability is at or above the model's cut-off, so this application is flagged."
    : "The default probability is below the model's cut-off, so this application passes.";
}

/* ---------- submit ---------- */
function buildPayload() {
  const f = form.elements;
  const income = parseInt(f.person_income.value, 10);
  const loan = parseInt(f.loan_amnt.value, 10);
  return {
    person_age: parseInt(f.person_age.value, 10),
    person_income: income,
    person_home_ownership: f.person_home_ownership.value,
    person_emp_length: parseFloat(f.person_emp_length.value),
    loan_intent: f.loan_intent.value,
    loan_grade: f.loan_grade.value,
    loan_amnt: loan,
    loan_int_rate: parseFloat(f.loan_int_rate.value),
    loan_percent_income: Math.round((loan / income) * 100) / 100,
    cb_person_default_on_file: f.cb_person_default_on_file.value,
    cb_person_cred_hist_length: parseInt(f.cb_person_cred_hist_length.value, 10),
  };
}

function validate() {
  let ok = true;
  form.querySelectorAll("input[type=number]").forEach((i) => {
    const bad = i.value === "" || Number.isNaN(Number(i.value)) || Number(i.value) < Number(i.min || -Infinity);
    i.classList.toggle("bad", bad);
    if (bad) ok = false;
  });
  return ok;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;

  if (!validate()) {
    errorBox.textContent = "Check the highlighted fields. Each needs a valid number.";
    errorBox.hidden = false;
    return;
  }

  btn.disabled = true;
  btn.classList.add("loading");
  btn.querySelector(".label").textContent = "Analysing…";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    showResult(await res.json());
    if (window.innerWidth <= 860) card.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    errorBox.textContent = `Could not get a prediction. ${err.message}. If the server just woke up on Render, wait a few seconds and try again.`;
    errorBox.hidden = false;
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.querySelector(".label").textContent = "Check default risk";
  }
});
