// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import {
    getFirestore,
    collection,
    addDoc,
    getDocs,
    query,
    orderBy,
    deleteDoc,
    doc
} from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

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
const tagsInput = document.getElementById("tags-input");
const tweetsContainer = document.getElementById("tweets-container");

// Function to create a new tweet container
function createTweetElement(tweetContent, tags, docId = null) {
    const postElement = document.createElement("div");
    postElement.className = "tweet";

    const contentParagraph = document.createElement("p");
    contentParagraph.textContent = tweetContent; // Preserve raw text
    contentParagraph.style.whiteSpace = "pre-wrap";

    const tagsContainer = document.createElement("div");
    tagsContainer.className = "tags";
    tags.forEach((tag) => {
        const tagElement = document.createElement("span");
        tagElement.textContent = `#${tag.trim()}`;
        tagsContainer.appendChild(tagElement);
    });

    const timestampSpan = document.createElement("span");
    timestampSpan.className = "timestamp";
    timestampSpan.textContent = new Date().toLocaleString();

    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-button";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
        if (confirm("Are you sure you want to delete this tweet?")) {
            if (docId) {
                // Delete from Firestore
                try {
                    await deleteDoc(doc(db, "posts", docId));
                    console.log("Tweet deleted:", docId);
                } catch (error) {
                    console.error("Error deleting tweet:", error);
                    alert("Failed to delete tweet.");
                }
            }
            postElement.remove(); // Remove from UI
        }
    });

    postElement.appendChild(contentParagraph);
    postElement.appendChild(tagsContainer);
    postElement.appendChild(timestampSpan);
    postElement.appendChild(deleteButton);

    return postElement;
}

// Post a new tweet
postButton.addEventListener("click", async () => {
    const tweetContent = tweetInput.value.trim();
    const tags = tagsInput.value.split(",").map((tag) => tag.trim()).filter(Boolean); // Parse tags

    if (!tweetContent) {
        alert("Tweet content cannot be empty!");
        return;
    }

    try {
        const docRef = await addDoc(collection(db, "posts"), {
            content: tweetContent,
            tags: tags,
            timestamp: new Date(),
        });
        console.log("Tweet posted with ID:", docRef.id);

        // Create a new container and add it to the UI immediately
        const newTweet = createTweetElement(tweetContent, tags, docRef.id);
        tweetsContainer.prepend(newTweet);

        tweetInput.value = ""; // Clear the input field
        tagsInput.value = ""; // Clear tags field
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
            const postElement = createTweetElement(post.content, post.tags || [], doc.id);
            tweetsContainer.appendChild(postElement);
        });
    } catch (error) {
        console.error("Error loading tweets:", error);
        alert("Failed to load tweets. Please refresh the page.");
    }
}

// Load posts when the DOM is ready
document.addEventListener("DOMContentLoaded", loadPosts);
