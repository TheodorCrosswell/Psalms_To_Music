// Define the structure of a search result object that matches the Python backend's output
interface SearchResult {
  start_index: number;
  end_index: number;
  similarity: number;
  words_matched: string[];
}

// Get references to the necessary HTML elements from the DOM
const searchButton = document.getElementById(
  "searchButton"
) as HTMLButtonElement;
const clearButton = document.getElementById("clearButton") as HTMLButtonElement;
const searchText = document.getElementById("searchText") as HTMLInputElement;
const resultsDiv = document.getElementById("results") as HTMLDivElement;

/**
 * Creates and displays a vertically aligned comparison of two sets of lyrics.
 * @param original_lyrics An array of words from the original sentence.
 * @param substituted_lyrics An array of words from the new sentence.
 * @param containerId The ID of the HTML element where the view will be rendered.
 */
function createLyricComparisonView(
  original_lyrics: string[],
  substituted_lyrics: string[],
  containerId: string
) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container with id "${containerId}" not found.`);
    return;
  }

  // Clear any previous content in the container
  container.innerHTML = "";

  const sentenceContainer = document.createElement("div");
  sentenceContainer.classList.add("lyric-sentence-container");
  // Set a CSS custom property for the number of columns in the grid
  sentenceContainer.style.setProperty(
    "--word-count",
    String(original_lyrics.length)
  );

  for (let i = 0; i < original_lyrics.length; i++) {
    const wordPairContainer = document.createElement("div");
    wordPairContainer.classList.add("word-pair-container");

    const originalWord = document.createElement("span");
    originalWord.classList.add("original-word");
    originalWord.textContent = original_lyrics[i];

    const substitutedWord = document.createElement("span");
    substitutedWord.classList.add("substituted-word");
    // Use the corresponding substituted word, or an empty string if it doesn't exist
    substitutedWord.textContent = substituted_lyrics[i] || "";

    wordPairContainer.appendChild(originalWord);
    wordPairContainer.appendChild(substitutedWord);
    sentenceContainer.appendChild(wordPairContainer);
  }

  container.appendChild(sentenceContainer);
}

/**
 * An asynchronous function to fetch search results from the backend API.
 */
const performSearch = async () => {
  // Get the search query from the input field
  const query = searchText.value;
  const scoreCutoff = 95; // You can make this configurable if you want

  // Check if the query is empty
  if (!query) {
    resultsDiv.innerHTML = "<p>Please enter a search term.</p>";
    return;
  }

  // Display a loading message while the search is in progress
  resultsDiv.innerHTML = "<p>Searching for results...</p>";

  try {
    // Construct the URL with query parameters
    const url = `/api/find_matches/kjv?searchText=${encodeURIComponent(
      query
    )}&scoreCutoff=${scoreCutoff}`;

    // Make an API call to the backend using fetch
    const response = await fetch(url);

    // Check if the request was successful
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON response
    const results: SearchResult[] = await response.json();

    // Clear the results area
    resultsDiv.innerHTML = "";

    // Check if any results were found
    if (results.length === 0) {
      resultsDiv.innerHTML = "<p>No matches found.</p>";
      return;
    }

    // Split the original user query into an array of words to be used as the top lyric row.
    const original_lyrics = query.trim().split(/\s+/);

    // Create and append a new element for each result
    results.forEach((result, index) => {
      // Create a container for the entire result item (similarity + lyric view)
      const resultElement = document.createElement("div");
      resultElement.classList.add("result-item");

      const similarity = document.createElement("p");
      similarity.textContent = `Similarity: ${result.similarity.toFixed(2)}%`;

      // Create a dedicated container for the lyric comparison view that the function can target
      const lyricViewContainer = document.createElement("div");
      const lyricViewId = `lyric-comparison-${index}`;
      lyricViewContainer.id = lyricViewId;

      resultElement.appendChild(similarity);
      resultElement.appendChild(lyricViewContainer);

      resultsDiv.appendChild(resultElement);

      // Call the new function to render the comparison view for this result
      // The substituted lyrics are the matched words from the API response
      if (original_lyrics.length === result.words_matched.length) {
        createLyricComparisonView(
          original_lyrics,
          result.words_matched,
          lyricViewId
        );
      } else {
        console.warn(
          "Word count mismatch between query and result. Displaying as plain text."
        );
        lyricViewContainer.innerHTML = `<p><strong>Original:</strong> ${original_lyrics.join(
          " "
        )}</p><p><strong>Matched:</strong> ${result.words_matched.join(
          " "
        )}</p>`;
      }
    });
  } catch (error) {
    console.error("Error fetching search results:", error);
    resultsDiv.innerHTML = "<p>An error occurred while fetching results.</p>";
  }
};

/**
 * Clears the search input and results.
 */
const clearSearch = () => {
  searchText.value = "";
  resultsDiv.innerHTML = "<p>Results will appear here.</p>";
};

// Add event listeners to the buttons
searchButton.addEventListener("click", performSearch);
clearButton.addEventListener("click", clearSearch);
