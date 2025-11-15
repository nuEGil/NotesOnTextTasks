document.getElementById("searchForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const query = document.getElementById("searchInput").value;

    const display = document.getElementById("contentDisplay");
    display.innerHTML = "<p><em>Loading...</em></p>";

    try {
        const response = await fetch("http://127.0.0.1:8000/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: query })
        });

        if (!response.ok) throw new Error("Server error: " + response.status);

        const data = await response.json();
        display.innerHTML = `<pre>${data.result}</pre>`;

    } catch (error) {
        display.innerHTML = `<p style="color:red;">${error}</p>`;
    }
});
