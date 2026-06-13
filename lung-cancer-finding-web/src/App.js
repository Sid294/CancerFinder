import React, { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // When user selects image or takes a camera picture
  function handleImageChange(event) {
    const file = event.target.files[0];

    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }

  }

  // Send image to FastAPI backend
  async function analyzeImage() {
    if (!image) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);

    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to connect to AI model.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="App">
      <h1>Cinder AI Cancer Detection</h1>

      {/* Camera / File input */}
      <input
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleImageChange}
      />

      {/* Show image preview */}
      {preview && (
        <div>
          <h3>Selected Image:</h3>
          <img
            src={preview}
            alt="Preview"
            width="300"
          />
        </div>
      )}

      <br />

      <button onClick={analyzeImage}>
        Analyze Image
      </button>

      {loading && <p>Running AI model...</p>}

      {/* Show prediction */}
      {result && (
        <div>
          <h2>Prediction Result</h2>
          <p>
            <strong>Status:</strong> {result.prediction}
          </p>

          <p>
            <strong>Confidence:</strong>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>

          <h3>Probabilities</h3>
          <p>
            Non-cancerous:{" "}
            {(result.probabilities?.["non-cancerous"] * 100).toFixed(2)}%
          </p>

          <p>
            Cancerous:{" "}
            {(result.probabilities?.["cancerous"] * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default App;