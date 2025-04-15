
document.addEventListener("DOMContentLoaded", function () {
    // Existing popup form code
    const openFormBtn = document.getElementById("openForm");
    const closeFormBtn = document.getElementById("closeForm");
    const popupForm = document.getElementById("popupForm");

    openFormBtn.addEventListener("click", function () {
        popupForm.style.display = "flex";
    });

    closeFormBtn.addEventListener("click", function () {
        popupForm.style.display = "none";
    });

    popupForm.addEventListener("click", function (event) {
        if (event.target === popupForm) {
            popupForm.style.display = "none";
        }
    });

    // Camera elements and functionality
    const cameraContainer = document.getElementById('camera-container');
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    let stream = null;

    // Start camera automatically when page loads
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment', // Prefer rear camera
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            cameraFeed.srcObject = stream;
        } catch (err) {
            console.error("Camera error:", err);
            showErrorModal("Could not access camera. Please check permissions.");
        }
    }

    // Capture image
    captureBtn.addEventListener('click', function() {
        if (!stream) return;

        const canvas = document.createElement('canvas');
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function(blob) {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            classifyImage(file, true);
        }, 'image/jpeg', 0.8);
    });

    // Upload functionality
    const uploadInput = document.getElementById("upload-input");
    uploadInput.addEventListener("change", function(e) {
        if (e.target.files.length > 0) {
            classifyImage(e.target.files[0], false);
        }
    });

    function classifyImage(file, isCamera) {
        showLoadingIndicator();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/classify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoadingIndicator();

            if (data.error) {
                throw new Error(data.error);
            }

            if (data.success) {
                showResultsModal(data, isCamera);
            } else {
                throw new Error('Classification failed');
            }
        })
        .catch(error => {
            hideLoadingIndicator();
            showErrorModal(error.message);
        });
    }

    function showLoadingIndicator() {
        // Create or show loading indicator
        let loader = document.getElementById('loading-indicator');
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'loading-indicator';
            loader.style.position = 'fixed';
            loader.style.top = '0';
            loader.style.left = '0';
            loader.style.width = '100%';
            loader.style.height = '100%';
            loader.style.backgroundColor = 'rgba(0,0,0,0.7)';
            loader.style.display = 'flex';
            loader.style.justifyContent = 'center';
            loader.style.alignItems = 'center';
            loader.style.zIndex = '1000';

            loader.innerHTML = `
                <div style="background: white; padding: 30px; border-radius: 10px; text-align: center;">
                    <h3>Processing Image...</h3>
                    <div class="spinner"></div>
                </div>
            `;

            document.body.appendChild(loader);
        } else {
            loader.style.display = 'flex';
        }
    }

    function hideLoadingIndicator() {
        const loader = document.getElementById('loading-indicator');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    function showResultsModal(data, isCamera) {
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'results-modal';
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.zIndex = '1000';

        // Determine color based on waste type
        const typeColor = data.waste_type === 'Recyclable' ? '#4CAF50' :
                         data.waste_type === 'Hazardous' ? '#F44336' : '#FF9800';

        // Modal content
        modal.innerHTML = `
            <div class="results-content" style="background: white; padding: 30px; border-radius: 15px; max-width: 500px; width: 90%;">
                <h2 style="color: #164A41; margin-bottom: 20px;">Classification Results</h2>

                ${data.image_url ? `
                    <div style="margin-bottom: 20px; text-align: center;">
                        <img src="${data.image_url}" style="max-width: 100%; max-height: 200px; border-radius: 8px;">
                    </div>
                ` : ''}

                <div style="margin-bottom: 15px;">
                    <strong>Category:</strong> ${data.category}
                </div>

                <div style="margin-bottom: 15px;">
                    <strong>Type:</strong>
                    <span style="color: ${typeColor}; font-weight: bold;">${data.waste_type}</span>
                </div>

                <div style="margin-bottom: 25px;">
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                </div>

                <div style="display: flex; justify-content: space-between;">
                    <button id="closeResults" style="padding: 10px 20px; background: #4D774E; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Close
                    </button>
                    <button id="viewDetails" style="padding: 10px 20px; background: #F1B24A; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        View Details
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add event listeners
        document.getElementById('closeResults').addEventListener('click', function() {
            document.body.removeChild(modal);
        });

        document.getElementById('viewDetails').addEventListener('click', function() {
            // Redirect to afterscan page with results
            window.location.href = `afterscan/afterscan.html?category=${encodeURIComponent(data.category)}&type=${encodeURIComponent(data.waste_type)}&confidence=${data.confidence}&image=${encodeURIComponent(data.image_url || '')}`;
        });
    }

    function showErrorModal(message) {
        const modal = document.createElement('div');
        modal.className = 'error-modal';
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.zIndex = '1000';

        modal.innerHTML = `
            <div style="background: white; padding: 30px; border-radius: 15px; max-width: 500px; width: 90%; text-align: center;">
                <h3 style="color: #F44336; margin-bottom: 20px;">Error</h3>
                <p style="margin-bottom: 25px;">${message}</p>
                <button id="closeError" style="padding: 10px 20px; background: #4D774E; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    OK
                </button>
            </div>
        `;

        document.body.appendChild(modal);

        document.getElementById('closeError').addEventListener('click', function() {
            document.body.removeChild(modal);
        });
    }

    // Start the camera when page loads
    startCamera();
});
