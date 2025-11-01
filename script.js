// Scroll to Top Button
const scrollBtn = document.getElementById("scrollBtn");

window.onscroll = function() {
  if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
    scrollBtn.style.display = "block";
  } else {
    scrollBtn.style.display = "none";
  }
};

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Simulated Cloud Prediction
const predictions = [
  "Tropical cloud cluster forming over Bay of Bengal — Probability: 82%",
  "No significant cluster detected — Clear tropical atmosphere",
  "Possible storm cloud detected near Indian Ocean — Confidence: 76%",
  "Heavy convective cloud build-up — Probability: 90%",
  "Stable weather conditions detected — Low storm probability"
];

function updatePrediction() {
  const element = document.getElementById("prediction-text");
  const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];
  element.innerText = randomPrediction;
}

// Auto update every 10 seconds
setInterval(updatePrediction, 10000);

// Initial display
document.addEventListener("DOMContentLoaded", updatePrediction);