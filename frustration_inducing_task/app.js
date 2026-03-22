let currentPage = 0;
const startTime = Date.now();

function loadPage(i) {
  const elapsed = (Date.now() - startTime) / 1000;
  const phase = elapsed < 300 ? "annoying" : "frustrating"; // gets worse over time
  const task = tasks[i];

  let html = "<form id='taskForm'><h2>" + task.title + "</h2>";

  switch(task.type) {
    case "text":
      task.questions.forEach((q, idx) => {
        html += `<label>${q}<input type="text" required></label><br>`;
      });
      break;
    case "textarea":
      html += `<label>${task.question}<br><textarea rows="6" cols="50" required></textarea></label>`;
      break;
    case "image":
      html += `<img src="${task.src}"><label>${task.question}<input type="text" required></label>`;
      break;
    case "radio":
      html += `<p>${task.question}</p>`;
      task.options.forEach(opt => html += `<label><input type="radio" name="q" required> ${opt}</label><br>`);
      break;
    case "checkbox":
      html += `<p>${task.question}</p>`;
      task.options.forEach(opt => html += `<label><input type="checkbox" name="q"> ${opt}</label><br>`);
      break;
    case "file":
      html += `<label>${task.question}<input type="file" required></label>`;
      break;
    case "slider":
      html += `<label>${task.question}<input type="range" min="0" max="100" value="50"></label>`;
      break;
    case "memory":
      html += `<p>Memorize these words for 5 seconds:</p><p>${task.words.join(", ")}</p>`;
      setTimeout(() => {
        document.querySelector("#content").innerHTML =
          "<form id='taskForm'><h2>" + task.title + "</h2>" +
          "<label>Recall the words:<textarea required></textarea></label>" +
          "<button type='submit'>Next</button></form>";
      }, 5000);
      break;
  }

  if (task.type !== "memory") {
    html += "<br><button type='submit'>Next</button></form>";
    document.getElementById("content").innerHTML = html;
  }

  document.getElementById("taskForm")?.addEventListener("submit", e => {
    e.preventDefault();
    frustrate(phase);
  });
}

function frustrate(phase) {
  const feedback = document.getElementById("feedback");

  if (phase === "annoying" || phase === "frustrating") {
    if (Math.random() < 0.3) { feedback.textContent = "⚠ Network error, try again."; return; }
    if (Math.random() < 0.2) { alert("System notice: Are you still there?"); }
    if (phase === "frustrating" && Math.random() < 0.25) {
      feedback.textContent = "⏳ Submitting... please wait.";
      document.getElementById("content").innerHTML += "<div class='spinner'></div>";
      setTimeout(() => nextPage(), 5000);
      return;
    }
  }
  feedback.textContent = "✔ Submitted!";
  nextPage();
}

function nextPage() {
  currentPage++;
  if (currentPage < tasks.length) {
    loadPage(currentPage);
  } else {
    document.getElementById("content").innerHTML = "<h2>✔ Task Completed</h2>";
  }
}

window.onload = () => loadPage(currentPage);
