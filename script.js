let video;
let model;
const canvas = document.getElementById('camera-canvas');
const ctx = canvas.getContext('2d');
let isMirroring = true;

// Function to toggle between different pages
function showPage(pageId) {
  const pages = document.querySelectorAll('.page');
  pages.forEach(page => page.classList.add('hidden'));
  document.getElementById(pageId).classList.remove('hidden');

  if (pageId === 'signPage') {
    startHandDetection();
  }
}

// Placeholder for hand detection result
function translateHandGesture(landmarks) {
  const thumbTip = landmarks[4];   // Thumb tip
  const indexTip = landmarks[8];   // Index finger tip
  const middleTip = landmarks[12]; // Middle finger tip
  const ringTip = landmarks[16];   // Ring finger tip
  const pinkyTip = landmarks[20];  // Pinky tip
  const wrist = landmarks[0];      // Wrist (for thumb distance check)

  // Check for "I Love You" gesture
  if (thumbTip[1] < indexTip[1] && pinkyTip[1] < indexTip[1]) {
    document.getElementById('translationResult').textContent = "I Love You";

  // Check for "Goodbye" gesture: thumb extended outwards, other four fingers together
  } else if (
    thumbTip[0] < wrist[0] - 50 && // Thumb is far to the left of the wrist (indicating it's extended)
    Math.abs(indexTip[0] - middleTip[0]) < 20 && // Index and middle fingers are close together
    Math.abs(middleTip[0] - ringTip[0]) < 20 &&  // Middle and ring fingers are close together
    Math.abs(ringTip[0] - pinkyTip[0]) < 20      // Ring and pinky fingers are close together
  ) {
    document.getElementById('translationResult').textContent = "Goodbye";
  } else if (
    indexTip[1] < middleTip[1] &&       // Index finger is higher than the middle finger
    indexTip[1] < ringTip[1] &&         // Index finger is higher than the ring finger
    indexTip[1] < pinkyTip[1] &&        // Index finger is higher than the pinky
    middleTip[1] > wrist[1] &&          // Middle finger is bent downward
    ringTip[1] > wrist[1] &&            // Ring finger is bent downward
    pinkyTip[1] > wrist[1] &&           // Pinky finger is bent downward
    thumbTip[0] < indexTip[0]           // Thumb is out of the way (positioned to the side or below)
  ) {
    document.getElementById('translationResult').textContent = "You";

  // Fallback for unknown gestures
  } else {
    document.getElementById('translationResult').textContent = "Unknown Gesture";
  }
}


// Initialize camera
async function setupCamera() {
  video = document.getElementById('camera');
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 400 } });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => resolve(video);
  });
}

// Detect hand
async function detectHand() {
  model = await handpose.load();

  setInterval(async () => {
    const predictions = await model.estimateHands(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw the video (in mirrored mode if enabled)
    if (isMirroring) {
      ctx.save();
      ctx.scale(-1, 1); // Flip horizontally
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();
    } else {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    if (predictions.length > 0) {
      predictions.forEach(prediction => {
        const landmarks = prediction.landmarks;
        translateHandGesture(landmarks);

        for (let i = 0; i < landmarks.length; i++) {
          const [x, y] = landmarks[i];
          ctx.beginPath();
          ctx.arc(isMirroring ? canvas.width - x : x, y, 6, 0, 4 * Math.PI);;
          ctx.fillStyle = "red";
          ctx.fill();
        }
      });
    }
  }, 100);
}

// Start hand detection
async function startHandDetection() {
  await setupCamera();
  video.play();
  detectHand();
}
// today 12/1
function toggleMirroring() {
  isMirroring = !isMirroring;
  alert(`Mirroring is now ${isMirroring ? "ON" : "OFF"}`);
}

// Feedback section
function submitFeedback() {
  const name = document.getElementById('userName').value;
  const email = document.getElementById('userEmail').value;
  const feedbackType = document.getElementById('feedbackType').value;
  const feedback = document.getElementById('userFeedback').value;
  const satisfaction = document.querySelector('input[name="satisfaction"]:checked')?.value;

  console.log("Feedback Submitted:", { name, email, feedbackType, feedback, satisfaction });
  alert("Thank you for your feedback!");
}

// Initial setup
showPage('signPage');

// =================== New Code ===================

// Map class labels to numbers
const labelMap = { "I_Love_You": 0, "Goodbye": 1, "Yes": 2 };

// Load images and preprocess them
async function loadImagesFromHTML() {
  const datasetDiv = document.getElementById("dataset"); // Get the dataset container
  const classDivs = datasetDiv.children; // Get all class-specific divs

  const imageTensors = [];
  const labels = [];

  for (const classDiv of classDivs) {
    const className = classDiv.getAttribute("data-class"); // Class name (e.g., "I_Love_You")
    const images = classDiv.querySelectorAll("img"); // Get all images for this class

    for (const img of images) {
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224]) // Resize to match model input size
        .toFloat()
        .div(tf.scalar(255.0)); // Normalize to [0, 1]

      imageTensors.push(tensor); // Add to image tensor array
      labels.push(labelMap[className]); // Add the corresponding label
    }
  }

  return {
    images: tf.stack(imageTensors), // Combine image tensors into one tensor
    labels: tf.oneHot(tf.tensor1d(labels, "int32"), Object.keys(labelMap).length) // One-hot encode labels
  };
}

// Train the model using HTML images
async function trainModelFromDataset() {
  // Load images and labels from HTML or folder
  const { images, labels } = await loadImagesFromHTML();

  // Create a simple CNN model
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [224, 224, 3], filters: 32, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({ optimizer: tf.train.adam(), loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  // Train the model
  await model.fit(images, labels, { epochs: 10, batchSize: 16 });
  console.log("Training complete!");

  // Save the trained model
  await model.save("localstorage://asl-model");
}
