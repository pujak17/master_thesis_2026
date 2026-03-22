let currentPage = 0;
const startTime = Date.now();

const AUTOSAVE_KEY = "experiment_autosave_v1";

const errorMessages = [
  "Are you sure you typed that correctly?",
  "Connection lost. Please retry.",
  "Unexpected validation error.",
  "Something went wrong, try again.",
  "Session expired. Reloading soon.",
  "System timeout. Please wait."
];

// Sends timestamped events to Python logger on port 5003.
// If Python is not running the fetch silently fails — no disruption to the task.
function logTaskEvent(eventName, detail = "") {
  const payload = {
    timestamp: Date.now() / 1000,          // matches Python time.time()
    event: eventName,
    detail: String(detail),
    task_index: currentPage,
    task_title: (window.tasks && window.tasks[currentPage])
                  ? window.tasks[currentPage].title || ""
                  : "",
    task_type: (window.tasks && window.tasks[currentPage])
                  ? window.tasks[currentPage].type || ""
                  : "",
    elapsed_seconds: ((Date.now() - startTime) / 1000).toFixed(1)
  };

  fetch("http://127.0.0.1:5003/task_event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }).catch(() => {});  // silently swallow network errors
}

function autosaveNow() {
  try {
    const payload = {
      savedAt: new Date().toISOString(),
      currentPage,
      data: window.collectedPeopleDetails || {}
    };
    localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(payload));
  } catch (e) {}
}

function restoreAutosave() {
  try {
    const raw = localStorage.getItem(AUTOSAVE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    window.collectedPeopleDetails = payload.data || {};
    if (typeof payload.currentPage === "number") currentPage = payload.currentPage;
  } catch (e) {}
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function buildPeopleDetailsTxt(peopleData) {
  const ts = new Date().toISOString();
  return [
    `Export time: ${ts}`,
    "",
    typeof peopleData === "string" ? peopleData : JSON.stringify(peopleData, null, 2)
  ].join("\n");
}

function generateCountries() {
  return [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon",
    "Canada", "Cape Verde", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macao",
    "Macedonia", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "Norway", "Oman",
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Swaziland", "Sweden", "Switzerland",
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
  ];
}

function resetExperiment() {
  try { localStorage.removeItem(AUTOSAVE_KEY); } catch (e) {}
  currentPage = 0;
  window.collectedPeopleDetails = {};
  logTaskEvent("experiment_reset");
  setFeedback("Reset done. Starting over…", "warning");
  loadPage(0);
}

window.addEventListener("load", () => {
  window.collectedPeopleDetails = window.collectedPeopleDetails || {};
  restoreAutosave();

  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (confirm("Reset everything and start over?")) resetExperiment();
    });
  }

  if (!Array.isArray(window.tasks)) {
    document.getElementById("content").innerHTML =
      "<p style='color:#d32f2f'>Error: tasks.js did not load or `window.tasks` is not an array.</p>";
    return;
  }

  logTaskEvent("experiment_start");
  loadPage(currentPage);
});

function triggerFakeError(callback) {
  const modal = document.getElementById("fakeErrorModal");
  const retryBtn = document.getElementById("fakeRetryBtn");
  let retryCount = 0;
  const maxRetries = 3;

  modal.style.display = "block";
  logTaskEvent("fake_error_shown");

  const newRetryBtn = retryBtn.cloneNode(true);
  retryBtn.parentNode.replaceChild(newRetryBtn, retryBtn);

  newRetryBtn.addEventListener("click", () => {
    retryCount++;
    const errorText = modal.querySelector("#errorText");
    logTaskEvent("fake_error_retry_clicked", `attempt_${retryCount}`);

    if (retryCount < maxRetries) {
      errorText.textContent = `Retry attempt ${retryCount}/${maxRetries - 1} failed. Trying again...`;
      newRetryBtn.disabled = true;
      setTimeout(() => (newRetryBtn.disabled = false), 1000);
    } else {
      modal.style.display = "none";
      errorText.textContent = "Unable to connect to server. Please retry.";
      if (callback) callback();
    }
  });
}

function triggerPopupConfirmation(callback, confirmCount = 0) {
  const modal = document.getElementById("confirmationModal");
  const confirmYes = document.getElementById("confirmYes");
  const confirmNo = document.getElementById("confirmNo");
  const confirmText = document.getElementById("confirmationText");

  modal.style.display = "block";
  logTaskEvent("confirmation_shown", `count_${confirmCount}`);

  if (confirmCount === 0) confirmText.textContent = "Are you sure you want to continue?";
  else if (confirmCount === 1) confirmText.textContent = "Please confirm one more time";
  else if (confirmCount === 2) confirmText.textContent = "Really sure?";
  else if (confirmCount === 3) confirmText.textContent = "Last confirmation required";
  else confirmText.textContent = "Absolutely certain?";

  const newYes = confirmYes.cloneNode(true);
  const newNo = confirmNo.cloneNode(true);
  confirmYes.parentNode.replaceChild(newYes, confirmYes);
  confirmNo.parentNode.replaceChild(newNo, confirmNo);

  newYes.addEventListener("click", () => {
    logTaskEvent("confirmation_yes", `count_${confirmCount}`);
    if (confirmCount < 4) {
      modal.style.display = "none";
      triggerPopupConfirmation(callback, confirmCount + 1);
    } else {
      modal.style.display = "none";
      if (callback) callback();
    }
  });

  newNo.addEventListener("click", () => {
    logTaskEvent("confirmation_no", `count_${confirmCount}`);
    modal.style.display = "none";
    if (callback) callback();
  });
}

function loadPage(i) {
  const elapsed = (Date.now() - startTime) / 1000;
  const phase = elapsed < 60 ? "calm" : (elapsed < 400 ? "annoying" : "frustrating");

  const task = window.tasks[i];
  if (!task) return;

  // Log every time a new task page is shown
  logTaskEvent("task_loaded", `phase:${phase}`);

  let progress = ((i + 1) / window.tasks.length) * 100;
  if (Math.random() < 0.10 && phase === "frustrating") {
    progress = progress * (0.3 + Math.random() * 0.2);
  }

  const progressBar = document.getElementById("progress");
  progressBar.style.width = Math.max(progress, 5) + "%";

  if (phase === "frustrating") {
    progressBar.style.background = "linear-gradient(90deg,#ff3d00,#ff9800)";
    document.querySelector(".container").classList.add("flicker-hard");
  } else if (phase === "annoying") {
    progressBar.style.background = "linear-gradient(90deg,#ffd600,#ff9800)";
  } else {
    progressBar.style.background = "linear-gradient(90deg,#4caf50,#2196f3)";
  }

  let html = `<form id="taskForm"><h2>${task.title}</h2>`;

  switch (task.type) {
    case "wait":
      html += `<p>${task.message || ""}</p>`;
      html += `<button type="submit" class="moving-btn" id="waitNext" style="display:none;">Next</button>`;
      break;

    case "normal-questions":
      html += `<p>${task.question || ""}</p>`;
      html += `<div id="normalQuestions"></div>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "text":
      (task.questions || []).forEach((q, idx) => {
        html += `<label class="annoy-q" data-q="${idx}">${q}<input type="text" class="answerField"></label>`;
      });
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "textarea":
      html += `<label>${task.question}<br><textarea class="answerField"></textarea></label>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "dropdown-number":
      html += `<p>${task.question}</p>`;
      html += `<label><select id="numberSelect" class="answerField" required><option value="">${task.placeholder}</option></select></label>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "dropdown-trap":
      html += `<p>${task.question}</p>`;
      html += `<label><select id="countrySelect" class="answerField" required><option value="">-- select a country --</option></select></label>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "radio":
      html += `<p>${task.question}</p>`;
      (task.options || []).forEach(opt => {
        html += `<label><input type="radio" name="q" value="${opt}"> ${opt}</label><br>`;
      });
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "captcha-clicks":
      html += `<p>${task.question}</p>`;
      html += `<div id="clickTargets"></div>`;
      html += `<p id="clickCounter" style="font-weight: bold; color: #ff9800;">0 / ${task.requiredClicks} clicked</p>`;
      html += `<p id="timeoutMessage" style="font-size: 12px; color: #999;"></p>`;
      html += `<button type="submit" class="moving-btn" disabled id="captchaSubmit">Verify</button>`;
      break;

    case "repetitive-clicks":
      html += `<p>${task.question}</p>`;
      html += `<button type="button" id="clickButton" style="padding: 20px 40px; font-size: 18px; margin-bottom: 15px;">Click Me (0/${task.requiredClicks})</button>`;
      html += `<p id="clickProgress" style="color: #ff9800; font-weight: bold;"></p>`;
      html += `<button type="submit" class="moving-btn" id="repetitiveSubmit" disabled>Submit</button>`;
      break;

    case "moving-slider":
      html += `<p>${task.question}</p>`;
      html += `<input type="range" id="movingSlider" min="0" max="100" value="50" style="width: 100%; margin: 20px 0;">`;
      html += `<p id="sliderDisplay">Current: 50</p>`;
      html += `<p id="sliderHint" style="font-size: 12px; color: #999;">Try to hold it steady...</p>`;
      html += `<button type="submit" class="moving-btn" id="sliderSubmit" disabled>Verify Value</button>`;
      break;

    case "typing-test":
      html += `<p>${task.question || "Type the following text exactly:"}</p>`;
      html += `<div class="typing-test-container">${task.text}</div>`;
      html += `<label>Type here:<br><input type="text" id="typingInput" class="answerField" placeholder="Type the text above..."></label>`;
      html += `<p id="typingAccuracy" style="font-size: 12px; color: #666;"></p>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;

    case "auto-submit-trap":
      html += `<p>${task.message}</p>`;
      html += `<div class="countdown-timer"><span id="countdown">8</span> seconds remaining...</div>`;
      html += `<button type="submit" class="moving-btn" id="autoSubmitBtn">Submit Now</button>`;
      break;

    case "infinite-annoying-loop":
      html += `<p style="color: #d32f2f; font-weight: bold;">Please enter the correct answer to proceed.</p>`;
      html += `<label><input id="infiniteInput" type="text" class="answerField" placeholder="Type the correct answer..." autocomplete="off"></label>`;
      html += `<p id="hintText" style="font-size: 12px; color: #666; margin-top: 5px;">Hint: It's a number...</p>`;
      html += `<button type="submit" class="moving-btn" id="infiniteSubmit">Verify</button>`;
      html += `<p id="attemptCounter" style="font-size: 12px; color: #ff9800; margin-top: 10px;"></p>`;
      break;

    case "popup-confirmation":
      html += `<p>${task.message}</p>`;
      html += `<button type="submit" class="moving-btn">Continue</button>`;
      break;

    case "ranking":
      html += `<p>${task.question}</p>`;
      html += `<div id="rankingContainer" style="display: flex; flex-direction: column; gap: 10px;"></div>`;
      html += `<button type="submit" class="moving-btn" id="rankingSubmit">Next</button>`;
      break;

    default:
      html += `<p>Unknown task type: ${task.type}</p>`;
      html += `<button type="submit" class="moving-btn">Next</button>`;
      break;
  }

  html += `</form>`;
  document.getElementById("content").innerHTML = html;

  // Setup dropdown-number
  if (task.type === "dropdown-number") {
    const sel = document.getElementById("numberSelect");
    for (let n = task.min; n <= task.max; n++) {
      const opt = document.createElement("option");
      opt.value = String(n);
      opt.textContent = String(n);
      sel.appendChild(opt);
    }
  }

  // Setup normal-questions + restore saved values
  if (task.type === "normal-questions") {
    const container = document.getElementById("normalQuestions");
    container.innerHTML = "";

    (task.fields || []).forEach(f => {
      const wrap = document.createElement("label");
      wrap.style.display = "block";
      wrap.style.margin = "10px 0";
      wrap.dataset.key = f.key;

      const labelText = document.createElement("div");
      labelText.textContent = f.label;
      labelText.style.marginBottom = "6px";

      let inputEl;

      if (f.kind === "select") {
        inputEl = document.createElement("select");
        inputEl.className = "answerField";
        if (f.required) inputEl.required = true;

        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "-- choose --";
        inputEl.appendChild(empty);

        (f.options || []).forEach(opt => {
          const o = document.createElement("option");
          o.value = opt;
          o.textContent = opt;
          inputEl.appendChild(o);
        });
      } else {
        inputEl = document.createElement("input");
        inputEl.className = "answerField";
        inputEl.type = (f.kind === "number") ? "number" : "text";
        if (f.placeholder) inputEl.placeholder = f.placeholder;
        if (f.required) inputEl.required = true;

        if (f.kind === "number") {
          if (typeof f.min === "number") inputEl.min = String(f.min);
          if (typeof f.max === "number") inputEl.max = String(f.max);
          if (typeof f.step === "number") inputEl.step = String(f.step);
        }
      }

      const saved = (window.collectedPeopleDetails.normalQuestions || {})[f.key];
      if (saved !== undefined && saved !== null) inputEl.value = String(saved);

      wrap.appendChild(labelText);
      wrap.appendChild(inputEl);
      container.appendChild(wrap);
    });
  }

  // Setup dropdown-trap
  if (task.type === "dropdown-trap") {
    const sel = document.getElementById("countrySelect");
    generateCountries().forEach(country => {
      const opt = document.createElement("option");
      opt.value = country;
      opt.textContent = country;
      sel.appendChild(opt);
    });

    if (task.resetOnScroll) {
      sel.addEventListener("scroll", () => {
        if (Math.random() < 0.3) {
          sel.value = "";
          logTaskEvent("dropdown_reset", "scroll");
        }
      });
    }

    if (task.resetOnHover) {
      sel.addEventListener("mouseover", () => {
        if (Math.random() < 0.2) {
          sel.value = "";
          logTaskEvent("dropdown_reset", "hover");
        }
      });
    }
  }

  // Setup wait timer
  if (task.type === "wait") {
    const nextBtn = document.getElementById("waitNext");
    setTimeout(() => (nextBtn.style.display = "block"), task.delay);
  }

  // Setup auto-submit trap
  if (task.type === "auto-submit-trap") {
    let timeRemaining = task.duration / 1000;
    const countdownEl = document.getElementById("countdown");

    const countdown = setInterval(() => {
      timeRemaining--;
      countdownEl.textContent = timeRemaining;

      if (timeRemaining <= 0) {
        clearInterval(countdown);
        logTaskEvent("auto_submit_fired", "timeout");
        document.getElementById("taskForm").dispatchEvent(new Event("submit"));
      }
    }, 1000);

    document.getElementById("taskForm").addEventListener("input", () => {
      if (task.resetCountdownOnInput) {
        clearInterval(countdown);
        timeRemaining = task.duration / 1000;

        const newCountdown = setInterval(() => {
          timeRemaining--;
          countdownEl.textContent = timeRemaining;

          if (timeRemaining <= 0) {
            clearInterval(newCountdown);
            logTaskEvent("auto_submit_fired", "timeout_after_input_reset");
            document.getElementById("taskForm").dispatchEvent(new Event("submit"));
          }
        }, 1000);
      }
    });
  }

  // Setup typing test
  if (task.type === "typing-test") {
    const input = document.getElementById("typingInput");
    const accuracyDisplay = document.getElementById("typingAccuracy");

    input.addEventListener("input", () => {
      const typed = input.value;
      const target = task.text;
      let correctChars = 0;

      for (let k = 0; k < typed.length && k < target.length; k++) {
        if (typed[k] === target[k]) correctChars++;
      }

      const accuracy = typed.length > 0 ? Math.round((correctChars / target.length) * 100) : 0;
      accuracyDisplay.textContent = `Accuracy: ${accuracy}%`;
    });
  }

  // CAPTCHA clicks
  if (task.type === "captcha-clicks") {
    const container = document.getElementById("clickTargets");
    const timeoutMessage = document.getElementById("timeoutMessage");
    let clickCount = 0;
    let buttonsMoving = true;

    setTimeout(() => {
      buttonsMoving = false;
      timeoutMessage.textContent = "Time extended. Buttons have stopped moving. Click them now!";
      timeoutMessage.style.color = "#2196f3";
      logTaskEvent("captcha_buttons_stopped");
      setFeedback("Buttons are now stationary. You have unlimited time to click them.", "success");
    }, 90000);

    for (let j = 0; j < task.requiredClicks; j++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "✓";

      const maxLeft = Math.max(0, container.offsetWidth - 70);
      const maxTop = Math.max(0, container.offsetHeight - 70);
      btn.style.left = Math.random() * maxLeft + "px";
      btn.style.top = Math.random() * maxTop + "px";

      btn.addEventListener("mouseover", () => {
        if (buttonsMoving) {
          const maxLeft2 = Math.max(0, container.offsetWidth - 70);
          const maxTop2 = Math.max(0, container.offsetHeight - 70);
          btn.style.left = Math.random() * maxLeft2 + "px";
          btn.style.top = Math.random() * maxTop2 + "px";
        }
      });

      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clickCount++;
        btn.style.opacity = "0.3";
        btn.disabled = true;
        document.getElementById("clickCounter").textContent = `${clickCount} / ${task.requiredClicks} clicked`;

        if (clickCount === task.requiredClicks) {
          document.getElementById("captchaSubmit").disabled = false;
          logTaskEvent("captcha_complete");
          setFeedback("All targets clicked! Ready to submit.", "success");
        }
      });

      container.appendChild(btn);
    }
  }

  // Repetitive clicks
  if (task.type === "repetitive-clicks") {
    const clickBtn = document.getElementById("clickButton");
    let clickCount = 0;

    clickBtn.addEventListener("click", () => {
      clickCount++;
      clickBtn.textContent = `Click Me (${clickCount}/${task.requiredClicks})`;
      document.getElementById("clickProgress").textContent =
        `You're ${((clickCount / task.requiredClicks) * 100).toFixed(0)}% done...`;

      if (clickCount === task.requiredClicks) {
        document.getElementById("repetitiveSubmit").disabled = false;
        logTaskEvent("repetitive_clicks_complete", `clicks:${clickCount}`);
        setFeedback("Great, now submit.", "success");
      }
    });
  }

  // Moving slider
  if (task.type === "moving-slider") {
    const movSlider = document.getElementById("movingSlider");
    movSlider.addEventListener("input", () => {
      document.getElementById("sliderDisplay").textContent = `Current: ${movSlider.value}`;
      if (Math.random() < 0.2) {
        movSlider.value = Math.floor(Math.random() * 100);
        document.getElementById("sliderHint").textContent = "It drifted! Try again...";
        logTaskEvent("slider_drifted", `value:${movSlider.value}`);
      }
    });

    movSlider.addEventListener("change", () => {
      const val = parseInt(movSlider.value, 10);
      if (Math.abs(val - task.targetValue) <= task.tolerance) {
        document.getElementById("sliderSubmit").disabled = false;
      }
    });
  }

  // Ranking (drag-drop)
  if (task.type === "ranking") {
    const container = document.getElementById("rankingContainer");
    let rankingOrder = [...(task.items || [])];

    function renderRanking() {
      container.innerHTML = "";
      rankingOrder.forEach((item, idx) => {
        const div = document.createElement("div");
        div.style.cssText = "display:flex; align-items:center; gap:10px; padding:10px; background:#f5f5f5; border-radius:5px; cursor:move;";
        div.draggable = true;
        div.dataset.index = idx;

        div.innerHTML = `<span style="font-weight:bold; color:#ff9800; min-width:30px;">${idx + 1}.</span><span>${item}</span>`;

        div.addEventListener("dragstart", (e) => {
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", idx);
        });

        div.addEventListener("dragover", (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          div.style.background = "#e0e0e0";
        });

        div.addEventListener("dragleave", () => {
          div.style.background = "#f5f5f5";
        });

        div.addEventListener("drop", (e) => {
          e.preventDefault();
          const fromIdx = parseInt(e.dataTransfer.getData("text/plain"), 10);
          const toIdx = idx;

          if (fromIdx !== toIdx) {
            const temp = rankingOrder[fromIdx];
            rankingOrder[fromIdx] = rankingOrder[toIdx];
            rankingOrder[toIdx] = temp;
            logTaskEvent("ranking_reordered", rankingOrder.join(">"));
            renderRanking();
          }
        });

        container.appendChild(div);
      });

      task.rankingResult = rankingOrder;
    }

    renderRanking();
  }

  document.getElementById("taskForm").addEventListener("submit", e => {
    e.preventDefault();
    logTaskEvent("form_submit_attempt");
    if (task.type === "wait") return nextPage();
    if (task.type === "popup-confirmation") return triggerPopupConfirmation(() => checkAnswer(task, phase, i));
    checkAnswer(task, phase, i);
  });

  triggerAnnoyances(phase);
}

// ========= Interruptions after 60s =========
function triggerAnnoyances(phase) {
  const btn = document.querySelector(".moving-btn");
  const inputs = document.querySelectorAll("input, textarea, select");

  if (phase === "calm") return;

  if (btn) {
    btn.addEventListener("mouseover", () => {
      if (Math.random() < 0.3) {
        btn.style.transform = `translate(${Math.random() * 40 - 20}px, ${Math.random() * 10 - 5}px)`;
      }
    });

    btn.addEventListener("click", e => {
      if (Math.random() < 0.3) {
        e.preventDefault();
        btn.disabled = true;
        logTaskEvent("button_blocked", "annoyance");
        setTimeout(() => {
          btn.disabled = false;
          setFeedback("Please try again.", "warning");
        }, 1200);
      }
    });
  }

  inputs.forEach(inp => {
    inp.addEventListener("input", () => {
      if ("value" in inp && typeof inp.value === "string" && Math.random() < 0.1) {
        inp.value = inp.value.slice(0, -1);
        logTaskEvent("input_deleted", "annoyance");
      }
    });
  });
}

// ========= Validation + storing answers + AUTOSAVE =========
function checkAnswer(task, phase, taskIndex) {
  if (phase === "frustrating" && Math.random() < 0.4) {
    triggerFakeError(() => checkAnswer(task, phase, taskIndex));
    return;
  }

  window.collectedPeopleDetails = window.collectedPeopleDetails || {};

  if (task.type === "text") {
    const fields = document.querySelectorAll(".answerField");
    const answers = [];

    for (let f of fields) {
      if (!f.value.trim()) {
        setFeedback("Please complete all fields.", "error");
        logTaskEvent("validation_failed", "empty_text_field");
        return;
      }
      answers.push(f.value.trim());
    }

    window.collectedPeopleDetails.text = window.collectedPeopleDetails.text || {};
    window.collectedPeopleDetails.text[task.title || `text_${taskIndex}`] = answers;
  }

  if (task.type === "normal-questions") {
    const container = document.getElementById("normalQuestions");
    if (!container) {
      setFeedback("Internal error: normalQuestions container missing.", "error");
      return;
    }

    window.collectedPeopleDetails.normalQuestions = window.collectedPeopleDetails.normalQuestions || {};
    (task.fields || []).forEach(f => {
      const lab = container.querySelector(`label[data-key="${f.key}"]`);
      const el = lab ? lab.querySelector("input, select, textarea") : null;
      const value = el ? el.value : "";

      if (f.required && !String(value).trim()) {
        setFeedback("Please complete all fields.", "error");
        logTaskEvent("validation_failed", `missing_field:${f.key}`);
        throw new Error("Missing required field");
      }

      window.collectedPeopleDetails.normalQuestions[f.key] =
        (f.kind === "number" && value !== "") ? Number(value) : value;
    });
  }

  if (task.type === "textarea") {
    const t = document.querySelector("textarea");
    if (!t.value.trim()) {
      setFeedback("Please complete the text area.", "error");
      logTaskEvent("validation_failed", "empty_textarea");
      return;
    }
    window.collectedPeopleDetails.textarea = window.collectedPeopleDetails.textarea || {};
    window.collectedPeopleDetails.textarea[task.title || `textarea_${taskIndex}`] = t.value.trim();
  }

  if (task.type === "dropdown-number") {
    const sel = document.getElementById("numberSelect");
    if (!sel.value) {
      setFeedback("Choose a number before continuing.", "error");
      logTaskEvent("validation_failed", "no_number_selected");
      return;
    }
    const picked = Number(sel.value);
    if (task.correct !== null && task.correct !== undefined && picked !== task.correct) {
      logTaskEvent("wrong_answer", `picked:${picked}_correct:${task.correct}`);
      setFeedback(errorMessages[Math.floor(Math.random() * errorMessages.length)], "error");
      return;
    }
    window.collectedPeopleDetails.dropdownNumber = window.collectedPeopleDetails.dropdownNumber || {};
    window.collectedPeopleDetails.dropdownNumber[task.title || `dropdownNumber_${taskIndex}`] = picked;
  }

  if (task.type === "dropdown-trap") {
    const sel = document.getElementById("countrySelect");
    if (!sel.value) {
      setFeedback("Please select a country.", "error");
      logTaskEvent("validation_failed", "no_country_selected");
      return;
    }
    window.collectedPeopleDetails.country = sel.value;
  }

  if (task.type === "radio") {
    const selected = document.querySelector("input[name='q']:checked");
    if (!selected) {
      setFeedback("Please select an option.", "error");
      logTaskEvent("validation_failed", "no_radio_selected");
      return;
    }
    if (task.correct && selected.value !== task.correct) {
      logTaskEvent("wrong_answer", `picked:${selected.value}_correct:${task.correct}`);
      setFeedback(errorMessages[Math.floor(Math.random() * errorMessages.length)], "error");
      return;
    }
    window.collectedPeopleDetails.radio = window.collectedPeopleDetails.radio || {};
    window.collectedPeopleDetails.radio[task.title || `radio_${taskIndex}`] = selected.value;
  }

  if (task.type === "captcha-clicks") {
    const clicksRecorded = parseInt(document.getElementById("clickCounter").textContent.split(" ")[0], 10);
    if (clicksRecorded < task.requiredClicks) {
      setFeedback("You must click all targets to proceed.", "error");
      logTaskEvent("validation_failed", `captcha_clicks:${clicksRecorded}/${task.requiredClicks}`);
      return;
    }
    window.collectedPeopleDetails.captchaClicks = { required: task.requiredClicks, done: clicksRecorded };
  }

  if (task.type === "repetitive-clicks") {
    const clickCount = parseInt(document.getElementById("clickButton").textContent.split("(")[1].split("/")[0], 10);
    if (clickCount < task.requiredClicks) {
      setFeedback(`You must click ${task.requiredClicks} times.`, "error");
      logTaskEvent("validation_failed", `rep_clicks:${clickCount}/${task.requiredClicks}`);
      return;
    }
    window.collectedPeopleDetails.repetitiveClicks = { required: task.requiredClicks, done: clickCount };
  }

  if (task.type === "moving-slider") {
    const sliderVal = parseInt(document.getElementById("movingSlider").value, 10);
    if (Math.abs(sliderVal - task.targetValue) > task.tolerance) {
      setFeedback(`Slider must be between ${task.targetValue - task.tolerance} and ${task.targetValue + task.tolerance}.`, "error");
      logTaskEvent("validation_failed", `slider:${sliderVal}_target:${task.targetValue}`);
      return;
    }
    window.collectedPeopleDetails.slider = sliderVal;
  }

  if (task.type === "ranking") {
    window.collectedPeopleDetails.ranking = window.collectedPeopleDetails.ranking || {};
    window.collectedPeopleDetails.ranking[task.title || `ranking_${taskIndex}`] = task.rankingResult || [];
  }

  if (task.type === "typing-test") {
    const input = document.getElementById("typingInput");
    const target = task.text || "";
    let correctChars = 0;

    for (let i = 0; i < input.value.length && i < target.length; i++) {
      if (input.value[i] === target[i]) correctChars++;
    }

    const accuracy = Math.round((correctChars / Math.max(1, target.length)) * 100);
    if (accuracy < task.minAccuracy) {
      setFeedback(`Accuracy too low (${accuracy}%). Must be at least ${task.minAccuracy}%.`, "error");
      logTaskEvent("validation_failed", `typing_accuracy:${accuracy}`);
      return;
    }
    window.collectedPeopleDetails.typingTest = { typed: input.value, accuracy };
  }

  if (task.type === "infinite-annoying-loop") {
    const inp = document.getElementById("infiniteInput");
    const counterEl = document.getElementById("attemptCounter");

    task.attempts = (task.attempts || 0) + 1;
    const remaining = task.maxAttempts - task.attempts;
    logTaskEvent("infinite_loop_attempt", `attempt:${task.attempts}_remaining:${remaining}`);

    if (Math.random() < 0.5) {
      setTimeout(() => {
        inp.value = "";
        logTaskEvent("infinite_loop_autoclear");
        setFeedback("Form was auto-cleared. Please retry.", "warning");
      }, task.resetEvery);
    }

    if (task.attempts < task.maxAttempts) {
      counterEl.textContent = `Incorrect. Attempts remaining: ${remaining}`;
      setFeedback("That's not quite right. Try again.", "error");
      return;
    }

    setFeedback("Okay, that's close enough. Moving on...", "warning");
    window.collectedPeopleDetails.infiniteLoop = { attempts: task.attempts };
    autosaveNow();
    logTaskEvent("infinite_loop_passed", `total_attempts:${task.attempts}`);
    setTimeout(nextPage, 1500);
    return;
  }

  autosaveNow();

  if (Math.random() < 0.2 && phase === "frustrating") {
    logTaskEvent("fake_submission_delay");
    setFeedback("Loading…", "warning");
    setTimeout(() => setFeedback("Please retry submission.", "error"), 2500);
    return;
  }

  frustrate();
}

// ========= Finish / next =========
function frustrate() {
  if (Math.random() < 0.25) {
    const msg = errorMessages[Math.floor(Math.random() * errorMessages.length)];
    logTaskEvent("fake_error_on_submit", msg);
    setFeedback(msg, "warning");
    return;
  }
  logTaskEvent("task_step_completed");
  setFeedback("Submitted successfully.", "success");
  setTimeout(nextPage, 800);
}

function nextPage() {
  if (Math.random() < 0.05) {
    logTaskEvent("fake_session_expired");
    setFeedback("Session expired. Restarting…", "error");
    setTimeout(() => location.reload(), 1500);
    return;
  }

  currentPage++;
  autosaveNow();

  if (currentPage < window.tasks.length) {
    loadPage(currentPage);
  } else {
    logTaskEvent("experiment_complete");
    document.getElementById("content").innerHTML = "<h2>Task Completed</h2>";
    setFeedback("All tasks finished.", "success");

    const txt = buildPeopleDetailsTxt(window.collectedPeopleDetails || {});
    downloadTextFile(`experiment_people_details_${Date.now()}.txt`, txt);
  }
}

// ========= Feedback UI =========
function setFeedback(msg, type = "info") {
  const f = document.getElementById("feedback");
  f.className = "";
  f.classList.add(
    type === "error" ? "feedback-error" :
    type === "warning" ? "feedback-warning" : "feedback-success"
  );
  f.textContent = msg;
}