async function predict() {
    const features = [
        parseFloat(document.getElementById('feature1').value),
        parseFloat(document.getElementById('feature2').value),
        parseFloat(document.getElementById('feature3').value),
        parseFloat(document.getElementById('feature4').value),
        parseFloat(document.getElementById('feature5').value),
        parseFloat(document.getElementById('feature6').value),
        parseFloat(document.getElementById('feature7').value)
    ];

    const response = await fetch('http://localhost:5000/predictRF', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
    });

    const result = await response.json();
    document.getElementById('result').textContent = 'Prediction: ' + result.prediction;
}