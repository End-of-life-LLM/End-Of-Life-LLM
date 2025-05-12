      let settings = {
            apiKey: "",
            temperature: 0.7,
            maxTokens: 1000,
            files: [],
            urls: [],
            timeout: 30
        };
        
        // Update sliders and their displayed values
        document.getElementById("temperature").addEventListener("input", function() {
            settings.temperature = parseFloat(this.value);
            document.getElementById("temperatureValue").textContent = settings.temperature;
        });
        
        document.getElementById("maxTokens").addEventListener("input", function() {
            settings.maxTokens = parseInt(this.value);
            document.getElementById("maxTokensValue").textContent = settings.maxTokens;
        });
        
        document.getElementById("timeout").addEventListener("input", function() {
            settings.timeout = parseInt(this.value);
            document.getElementById("timeoutValue").textContent = settings.timeout;
        });
        
        // Handle API key
        document.getElementById("apiKey").addEventListener("change", function() {
            settings.apiKey = this.value;
        });
        
        // Handle file upload
        document.getElementById("fileUpload").addEventListener("change", function() {
            settings.files = this.files;
            document.getElementById("fileCount").textContent = settings.files.length;
        });
        
        // Handle URLs
        document.getElementById("urls").addEventListener("change", function() {
            settings.urls = this.value.split("\n").filter(url => url.trim() !== "");
        });
        
        // Navigate to home
        document.getElementById("homeButton").addEventListener("click", function() {
            window.location.href = '/home';
        });
        
        // Save settings
        document.getElementById("saveButton").addEventListener("click", function() {
            // Create FormData object to handle file uploads
            const formData = new FormData();
            
            // Add all settings to FormData
            formData.append("apiKey", document.getElementById("apiKey").value);
            formData.append("temperature", document.getElementById("temperature").value);
            formData.append("maxTokens", document.getElementById("maxTokens").value);
            formData.append("timeout", document.getElementById("timeout").value);
            formData.append("urls", document.getElementById("urls").value);
            
            // Add files to FormData
            const fileInput = document.getElementById("fileUpload");
            if (fileInput.files.length > 0) {
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append("files[]", fileInput.files[i]);
                }
            }
            
            // Send data to server
            fetch('/save_api_settings', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Show success message
                const statusEl = document.getElementById("statusMessage");
                statusEl.textContent = data.message;
                statusEl.className = "status-message success";
                statusEl.style.display = "block";
                
                // Save to localStorage as well (except files)
                localStorage.setItem("apiSettings", JSON.stringify({
                    apiKey: document.getElementById("apiKey").value,
                    temperature: parseFloat(document.getElementById("temperature").value), 
                    maxTokens: parseInt(document.getElementById("maxTokens").value),
                    urlCount: document.getElementById("urls").value.split("\n").filter(url => url.trim() !== "").length,
                    fileCount: fileInput.files.length,
                    timeout: parseInt(document.getElementById("timeout").value)
                }));
                
                console.log("Settings saved:", data);
            })
            .catch(error => {
                // Show error message
                const statusEl = document.getElementById("statusMessage");
                statusEl.textContent = "Error saving settings: " + error.message;
                statusEl.className = "status-message error";
                statusEl.style.display = "block";
                
                console.error("Error saving settings:", error);
            });
        });
        
        // Load settings from localStorage if available
        window.addEventListener("load", function() {
            const savedSettings = localStorage.getItem("apiSettings");
            if (savedSettings) {
                const parsed = JSON.parse(savedSettings);
                
                if (parsed.apiKey) {
                    document.getElementById("apiKey").value = parsed.apiKey;
                    settings.apiKey = parsed.apiKey;
                }
                
                if (parsed.temperature) {
                    document.getElementById("temperature").value = parsed.temperature;
                    document.getElementById("temperatureValue").textContent = parsed.temperature;
                    settings.temperature = parsed.temperature;
                }
                
                if (parsed.maxTokens) {
                    document.getElementById("maxTokens").value = parsed.maxTokens;
                    document.getElementById("maxTokensValue").textContent = parsed.maxTokens;
                    settings.maxTokens = parsed.maxTokens;
                }
                
                if (parsed.timeout) {
                    document.getElementById("timeout").value = parsed.timeout;
                    document.getElementById("timeoutValue").textContent = parsed.timeout;
                    settings.timeout = parsed.timeout;
                }
            }
        });