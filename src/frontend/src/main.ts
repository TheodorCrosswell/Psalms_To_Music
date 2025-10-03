// Define the type for a single word detail to ensure type safety
interface WordDetail {
  word_long: string;
  word_short: string;
  syllable_long: string;
  syllable_short: string;
}

// Get references to the HTML elements
const searchButton = document.getElementById(
  "searchButton"
) as HTMLButtonElement;
const clearButton = document.getElementById("clearButton") as HTMLButtonElement;
const searchText = document.getElementById("searchText") as HTMLInputElement;
const resultsDiv = document.getElementById("results") as HTMLDivElement;

// Store the initial state of the results div
const initialResultsHTML = resultsDiv.innerHTML;

/**
 * Renders the search results in the resultsDiv.
 * @param results - A nested array of WordDetail objects.
 */
const displayResults = (results: WordDetail[][]) => {
  // Clear previous results
  resultsDiv.innerHTML = "";

  if (results.length === 0) {
    resultsDiv.innerHTML = "<p>No results found.</p>";
    return;
  }

  // Create a container for all result groups
  const container = document.createElement("div");

  // Iterate over the outer array (each list is a match)
  results.forEach((resultGroup) => {
    const groupElement = document.createElement("div");
    groupElement.style.marginBottom = "10px"; // Add some space between groups

    // Join the words and syllables of the inner array to form a single string
    const resultText = resultGroup
      .map(
        (detail) =>
          `${detail.word_long} - ${detail.word_short} / ${detail.syllable_long} - ${detail.syllable_short} <br>`
      )
      .join(" ");

    const p = document.createElement("p");
    p.innerHTML = resultText;
    groupElement.appendChild(p);
    container.appendChild(groupElement);
  });

  resultsDiv.appendChild(container);
};

/**
 * Fetches search results from the API.
 */
const performSearch = async () => {
  const query = searchText.value.trim();
  if (!query) {
    resultsDiv.innerHTML = "<p>Please enter a search term.</p>";
    return;
  }

  resultsDiv.innerHTML = "<p>Searching...</p>";

  try {
    // The base URL of your FastAPI application
    const fastApiBaseUrl = "http://127.0.0.1:8000"; // IMPORTANT: Change this to your actual API address if it's different
    const response = await fetch(`${fastApiBaseUrl}/fuzzy_search/kjv/${query}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: WordDetail[][] = await response.json();
    displayResults(data);
  } catch (error) {
    console.error("Error fetching search results:", error);
    resultsDiv.innerHTML =
      "<p>Error fetching results. Please check the console for details.</p>";
  }
};

/**
 * Clears the search input and results.
 */
const clearSearch = () => {
  searchText.value = "";
  resultsDiv.innerHTML = initialResultsHTML;
};

// Add event listeners to the buttons
searchButton.addEventListener("click", performSearch);
clearButton.addEventListener("click", clearSearch);

// Optional: Allow pressing Enter in the input field to trigger a search
searchText.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    performSearch();
  }
});
