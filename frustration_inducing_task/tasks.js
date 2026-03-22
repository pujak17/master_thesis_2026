window.tasks = [
  {
    title: "Welcome",
    type: "wait",
    message: "Welcome. Click next when it appears.",
    delay: 10000
  },

  {
    title: "Information",
    type: "text",
    questions: [
      "Please enter your first name:",
      "Re-enter your first name to confirm:",
      "If you sometimes use a nickname, enter it here:",
      "Enter your name as you would prefer it to appear in official documents:"
    ]
  },

  {
    title: "Quick routine questions",
    type: "normal-questions",
    question: "Please answer these quick questions:",
    fields: [
      { key: "workHours", label: "How many hours do you work in a day?", kind: "number", min: 0, max: 24, step: 0.5, placeholder: "e.g. 8", required: true },
      { key: "breakMinutes", label: "How many minutes of breaks do you take per day?", kind: "number", min: 0, max: 300, step: 5, placeholder: "e.g. 45", required: true },
      { key: "commuteMinutes", label: "How long is your daily commute (minutes)?", kind: "number", min: 0, max: 300, step: 5, placeholder: "e.g. 30", required: true },
      { key: "workMode", label: "How do you usually work?", kind: "select", required: true, options: ["Remote", "Hybrid", "On-site", "Varies"] },
      { key: "startTime", label: "What time do you usually start work? (approx.)", kind: "text", placeholder: "e.g. 09:00", required: true },
      { key: "productivity", label: "When are you most productive?", kind: "select", required: true, options: ["Morning", "Afternoon", "Evening", "It depends"] },
      { key: "coffee", label: "How many coffees/teas do you typically have per day?", kind: "number", min: 0, max: 20, step: 1, placeholder: "e.g. 2", required: true },
      { key: "music", label: "Do you listen to music while working?", kind: "select", required: true, options: ["Never", "Sometimes", "Often", "Always"] },
      { key: "devices", label: "Main device you use most:", kind: "select", required: true, options: ["Laptop", "Desktop", "Tablet", "Phone", "Multiple"] },
      { key: "notes", label: "Anything you want to add? (optional)", kind: "text", placeholder: "Optional", required: false }
    ]
  },

  {
    title: "Personal Information",
    type: "radio",
    question: "What is your highest level of education?",
    options: ["High School", "Bachelor's Degree", "Master's Degree", "PhD"],
    correct: null
  },

  {
    title: "Experience Level",
    type: "dropdown-number",
    question: "How many years of experience do you have in your field?",
    min: 0,
    max: 60,
    correct: null,
    placeholder: "-- select years --"
  },

  {
    title: "Verification Quiz",
    type: "dropdown-number",
    question: "What is 25 multiplied by 2?",
    min: 0,
    max: 700,
    correct: 50,
    placeholder: "-- pick a number --"
  },

  {
    title: "Country Selection",
    type: "dropdown-trap",
    question: "Select your country:",
    resetOnScroll: true,
    resetOnHover: true
  },

  {
    title: "Typing Proficiency",
    type: "typing-test",
    text: "The quick brown fox jumps over the lazy dog",
    timeLimit: 30,
    minAccuracy: 98
  },

  {
    title: "Quick Assessment",
    type: "auto-submit-trap",
    message: "Form auto-submits in 8 seconds...",
    countdown: true,
    resetCountdownOnInput: true,
    duration: 8000
  },

  {
    title: "System Check",
    type: "wait",
    delay: 1500,
    message: "Please wait. This cannot be skipped."
  },

  {
    title: "Logic Assessment",
    type: "textarea",
    question: "If you try to fail and you succeed, did you fail or succeed? Explain your logic, but don't use the words 'fail' or 'succeed':"
  },

  {
    title: "Human Verification",
    type: "captcha-clicks",
    question: "Click on all the buttons to verify you are human.",
    requiredClicks: 5
  },

  {
    title: "Identity Confirmation",
    type: "repetitive-clicks",
    question: "Click the button exactly 10 times:",
    requiredClicks: 10
  },

  {
    title: "Precision Calibration",
    type: "moving-slider",
    question: "Move the slider to exactly 50:",
    targetValue: 50,
    tolerance: 2
  },

  {
    title: "Rank Preferences",
    type: "ranking",
    question: "Drag to rank these from most to least important:",
    items: ["Career", "Family", "Health", "Hobbies", "Friends"]
  },

  {
    title: "Confirm Action",
    type: "popup-confirmation",
    message: "Are you sure you want to continue?",
    maxPopups: 5
  },

  {
    title: "Final Processing",
    type: "wait",
    delay: 3000,
    message: "Almost done... please wait."
  },

  {
    title: "Security Verification",
    type: "infinite-annoying-loop",
    attempts: 0,
    maxAttempts: 7,
    resetEvery: 2000
  },

  {
    title: "Feedback Survey",
    type: "radio",
    question: "If nothing on this form was important, why did you waste time filling it out?",
    options: ["This form was important", "It wasn't important", "You're right, this is frustrating", "I refuse to answer"],
    correct: "You're right, this is frustrating"
  }
];
