document.addEventListener('DOMContentLoaded', () => {
    Chart.defaults.color = '#e0e0e0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';

    const companyList = document.getElementById('company-list');
    const messageArea = document.getElementById('message-area');
    const chartCanvas = document.getElementById('stock-chart');
    const chartTitle = document.getElementById('chart-title');
    const predictButton = document.getElementById('predict-button');
    const forecastTable = document.getElementById('forecast-table');
    const sentimentOverall = document.getElementById('sentiment-overall');
    const headlinesList = document.getElementById('headlines-list');
    const high52WeekElem = document.getElementById('high-52-week');
    const low52WeekElem = document.getElementById('low-52-week');
    const avgVolumeElem = document.getElementById('avg-volume');
        // Technical indicators panel elements
        const rsiElem = document.getElementById('rsi-value');
        const macdElem = document.getElementById('macd-value');
        const bbElem = document.getElementById('bb-value');

    let stockChart;
    let currentSymbol = null;
    let historicalData = [];

    const fetchCompanies = async () => {
        try {
            const response = await fetch('/companies');
            if (!response.ok) throw new Error(`Failed to fetch companies`);
            const companies = await response.json();
            companyList.innerHTML = ''; 
            companies.forEach(company => {
                const li = document.createElement('li');
                const button = document.createElement('button');
                button.textContent = company.name;
                button.addEventListener('click', () => {
                    fetchStockData(company.symbol, company.name, button);
                    fetchSentimentData(company.symbol);
                    fetchIndicatorData(company.symbol); // Fetch new stats
                });
                li.appendChild(button);
                companyList.appendChild(li);
            });
        } catch (error) {
            messageArea.textContent = 'Error: Could not load company list.';
        }
    };

    const fetchStockData = async (symbol, name, clickedButton) => {
        messageArea.textContent = `Fetching stock data for ${symbol}...`;
        chartTitle.textContent = `Stock Chart: ${name}`;
        currentSymbol = symbol;
        predictButton.disabled = true;
        forecastTable.innerHTML = '';

        document.querySelectorAll('#company-list button').forEach(btn => btn.classList.remove('active'));
        if (clickedButton) clickedButton.classList.add('active');

        try {
            const response = await fetch(`/stock/${symbol}`);
            if (!response.ok) throw new Error((await response.json()).detail);
            historicalData = await response.json();
            if (historicalData.length === 0) throw new Error("No historical data.");
            
            messageArea.textContent = '';
            renderChart(historicalData, symbol);
            predictButton.disabled = false;

                    // Show latest technical indicators
                    const latest = historicalData[historicalData.length - 1];
                    rsiElem.textContent = latest.RSI !== undefined && latest.RSI !== null ? latest.RSI.toFixed(2) : '-';
                    macdElem.textContent = latest.MACD !== undefined && latest.MACD !== null ? latest.MACD.toFixed(2) : '-';
                    if (latest.BB_Upper !== undefined && latest.BB_Upper !== null && latest.BB_Lower !== undefined && latest.BB_Lower !== null) {
                        bbElem.textContent = `Upper: ${latest.BB_Upper.toFixed(2)}, Lower: ${latest.BB_Lower.toFixed(2)}`;
                    } else {
                        bbElem.textContent = '-';
                    }
        } catch (error) {
            messageArea.textContent = `Error: ${error.message}`;
            if (stockChart) stockChart.destroy();
                    rsiElem.textContent = '-';
                    macdElem.textContent = '-';
                    bbElem.textContent = '-';
        }
    };
    
    const fetchIndicatorData = async (symbol) => {
        try {
            const response = await fetch(`/stock/${symbol}/indicators`);
            if (!response.ok) throw new Error('Indicators not found');
            const indicators = await response.json();
            high52WeekElem.textContent = `$${indicators.high_52_week.toLocaleString()}`;
            low52WeekElem.textContent = `$${indicators.low_52_week.toLocaleString()}`;
            avgVolumeElem.textContent = indicators.average_volume.toLocaleString();
        } catch (error) {
            high52WeekElem.textContent = '-';
            low52WeekElem.textContent = '-';
            avgVolumeElem.textContent = '-';
        }
    };

    const fetchSentimentData = async (symbol) => {
        sentimentOverall.textContent = 'Loading...';
        sentimentOverall.className = 'sentiment-neutral';
        headlinesList.innerHTML = '';
        try {
            const response = await fetch(`/sentiment/${symbol}`);
            if (!response.ok) throw new Error((await response.json()).detail);
            const sentimentData = await response.json();
            sentimentOverall.textContent = sentimentData.overall_sentiment;
            sentimentOverall.className = `sentiment-${sentimentData.overall_sentiment.toLowerCase()}`;
            if (sentimentData.articles.length === 0) {
                headlinesList.innerHTML = '<li>No recent news found.</li>';
            } else {
                sentimentData.articles.forEach(article => {
                    const li = document.createElement('li');
                    li.textContent = article.title;
                    li.className = article.sentiment.toLowerCase();
                    headlinesList.appendChild(li);
                });
            }
        } catch (error) {
            sentimentOverall.textContent = 'Error';
        }
    };

    const fetchPrediction = async () => {
        if (!currentSymbol) return;
        predictButton.disabled = true;
        predictButton.textContent = 'Calculating...';
        messageArea.textContent = 'Training model and generating forecast...';
        try {
            const response = await fetch(`/predict/${currentSymbol}`);
            if (!response.ok) throw new Error((await response.json()).detail);
            const result = await response.json();
            updateChartWithPredictions(result.forecast, result.backtest);
            populateForecastTable(result.forecast);
            messageArea.textContent = 'Forecast and backtest generated successfully.';
        } catch (error) {
            messageArea.textContent = `Error: ${error.message}`;
        } finally {
            predictButton.disabled = false;
            predictButton.textContent = "Generate Forecast & Backtest";
        }
    };

    const renderChart = (data, symbol) => {
        if (stockChart) stockChart.destroy();
        const dates = data.map(item => item.Date);
        const closingPrices = data.map(item => item.Close);
        const sma50 = data.map(item => item.SMA_50);
        const volumes = data.map(item => item.Volume);
        const ctx = chartCanvas.getContext('2d');
        stockChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: dates,
                datasets: [
                    {
                        type: 'line',
                        label: `${symbol} Closing Price`,
                        data: closingPrices,
                        borderColor: 'rgb(0, 123, 255)',
                        yAxisID: 'y',
                    },
                    {
                        type: 'line',
                        label: '50-Day SMA',
                        data: sma50,
                        borderColor: 'rgb(255, 159, 64)',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'y',
                    },
                    {
                        type: 'bar',
                        label: 'Volume',
                        data: volumes,
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'month' } },
                    y: { position: 'left', title: { display: true, text: 'Price (USD)' } },
                    y1: { position: 'right', title: { display: true, text: 'Volume' }, grid: { drawOnChartArea: false } },
                }
            }
        });
    };

    const updateChartWithPredictions = (forecast, backtest) => {
        // This function remains the same as before
        if (!stockChart) return;
        const allLabels = [...historicalData.map(d => d.Date), ...forecast.dates];
        const backtestData = Array(historicalData.length).fill(null);
        backtest.dates.forEach((date, index) => {
            const pos = historicalData.findIndex(d => d.Date === date);
            if (pos !== -1) backtestData[pos] = backtest.predicted_prices[index];
        });
        const forecastData = Array(historicalData.length).fill(null).concat(forecast.predicted_prices);
        const confLowerData = Array(historicalData.length).fill(null).concat(forecast.conf_lower);
        const confUpperData = Array(historicalData.length).fill(null).concat(forecast.conf_upper);
        stockChart.data.labels = allLabels;
        stockChart.data.datasets.push({ label: 'Backtest Prediction', data: backtestData, borderColor: 'rgb(255, 159, 64)', tension: 0.1, pointRadius: 0, yAxisID: 'y' }, { label: 'Forecast', data: forecastData, borderColor: 'rgb(255, 99, 132)', borderDash: [5, 5], tension: 0.1, yAxisID: 'y' }, { label: 'Confidence Interval', data: confUpperData, fill: '+1', backgroundColor: 'rgba(255, 99, 132, 0.2)', borderColor: 'transparent', pointRadius: 0, yAxisID: 'y' }, { label: 'Confidence Interval Lower', data: confLowerData, fill: false, borderColor: 'transparent', pointRadius: 0, yAxisID: 'y' });
        stockChart.update();
    };

    const populateForecastTable = (forecast) => {
        // This function remains the same as before
        let tableHTML = `<thead><tr><th>Date</th><th>Predicted Price</th><th>Range (95% Confidence)</th></tr></thead><tbody>`;
        forecast.dates.forEach((date, i) => {
            tableHTML += `<tr><td>${date}</td><td>$${forecast.predicted_prices[i]}</td><td>$${forecast.conf_lower[i]} - $${forecast.conf_upper[i]}</td></tr>`;
        });
        tableHTML += `</tbody>`;
        forecastTable.innerHTML = tableHTML;
    };
    
    predictButton.addEventListener('click', fetchPrediction);
    fetchCompanies();
});
  
