// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import { getFirestore, collection, addDoc, getDocs, query, orderBy } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyAe_zfT25zPIY6FevnY9P1ZlhE9f7RtLhM",
    authDomain: "uniproj-ccbb3.firebaseapp.com",
    databaseURL: "https://uniproj-ccbb3-default-rtdb.firebaseio.com",
    projectId: "uniproj-ccbb3",
    storageBucket: "uniproj-ccbb3.appspot.com",
    messagingSenderId: "330789578341",
    appId: "1:330789578341:web:00d3070e129edba44e088c",
    measurementId: "G-1SXYY50G9T",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// DOM Elements
const postButton = document.getElementById("post-tweet");
const tweetInput = document.getElementById("tweet-input");
const tweetsContainer = document.getElementById("tweets-container");

// Function to create a new tweet container
function createTweetElement(tweetContent) {
    const postElement = document.createElement("div");
    postElement.className = "tweet";
    postElement.innerHTML = `
        <p>${tweetContent}</p>
        <span class="timestamp">${new Date().toLocaleString()}</span>
    `;
    return postElement;
}

// Post a new tweet
postButton.addEventListener("click", async () => {
    const tweetContent = tweetInput.value.trim();

    if (!tweetContent) {
        alert("Tweet content cannot be empty!");
        return;
    }

    try {
        const docRef = await addDoc(collection(db, "posts"), {
            content: tweetContent,
            timestamp: new Date(),
            
        });
        console.log("Tweet posted with ID:", docRef.id);

        // Create a new container and add it to the UI immediately
        const newTweet = createTweetElement(tweetContent);
        tweetsContainer.prepend(newTweet); // Add new tweet at the top

        tweetInput.value = ""; // Clear the input field
    } catch (error) {
        console.error("Error posting tweet:", error);
        alert("Failed to post tweet. Please try again.");
    }
});

// Load all tweets from Firestore
async function loadPosts() {
    tweetsContainer.innerHTML = ""; // Clear existing tweets

    try {
        const postsQuery = query(collection(db, "posts"), orderBy("timestamp", "desc"));
        const querySnapshot = await getDocs(postsQuery);

        querySnapshot.forEach((doc) => {
            const post = doc.data();
            const postElement = createTweetElement(post.content);
            tweetsContainer.appendChild(postElement);
        });
    } catch (error) {
        console.error("Error loading tweets:", error);
        alert("Failed to load tweets. Please refresh the page.");
    }
}

// Load posts when the DOM is ready
document.addEventListener("DOMContentLoaded", loadPosts);
