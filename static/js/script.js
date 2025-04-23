async function predict() {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Predicting...';
    resultDiv.classList.remove('error');

    // Collect input values
    const data = {
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        pressure: document.getElementById('pressure').value,
        wind_speed: document.getElementById('wind_speed').value,
        precipitation: document.getElementById('precipitation').value,
        hour: document.getElementById('hour').value,
        day_of_week: document.getElementById('day_of_week').value
    };

    try {
        // Send POST request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.status === 'success') {
            resultDiv.textContent = `Predicted Temperature: ${result.prediction}°C`;
        } else {
            resultDiv.textContent = `Error: ${result.message}`;
            resultDiv.classList.add('error');
        }
    } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.classList.add('error');
    }
}