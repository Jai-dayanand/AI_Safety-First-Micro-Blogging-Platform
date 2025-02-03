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
    storageBucket: "uniproj-ccbb3.firebasestorage.app",
    messagingSenderId: "330789578341",
    appId: "1:330789578341:web:00d3070e129edba44e088c",
    measurementId: "G-1SXYY50G9T"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// DOM Elements
const newTweetButton = document.getElementById("new-tweet-button");
const tweetPopup = document.getElementById("tweet-popup");
const closePopup = document.getElementById("close-popup");
const postButton = document.getElementById("post-tweet");
const tweetInput = document.getElementById("tweet-input");
const tagsInput = document.getElementById("tags-input");
const tweetsContainer = document.getElementById("tweets-container");

const viewTweetPopup = document.getElementById("view-tweet-popup");
const closeViewPopup = document.getElementById("close-view-popup");
const viewTweetContent = document.getElementById("view-tweet-content");
const viewTweetTags = document.getElementById("view-tweet-tags");
const viewTweetTimestamp = document.getElementById("view-tweet-timestamp");

// Open popup for new tweet
newTweetButton.addEventListener("click", () => {
    tweetPopup.style.display = "flex";
});

// Close new tweet popup
closePopup.addEventListener("click", () => {
    tweetPopup.style.display = "none";
});

// Close view tweet popup
closeViewPopup.addEventListener("click", () => {
    viewTweetPopup.style.display = "none";
});

// Create a tweet element
function createTweetElement(tweetContent, tags, timestamp, docId = null) {
    const postElement = document.createElement("div");
    postElement.className = "tweet";

    const contentParagraph = document.createElement("p");
    contentParagraph.textContent = tweetContent;
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
    timestampSpan.textContent = new Date(timestamp).toLocaleString();

    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-button";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async (e) => {
        e.stopPropagation(); // Prevent the tweet container click event
        if (confirm("Are you sure you want to delete this tweet?")) {
            if (docId) {
                try {
                    await deleteDoc(doc(db, "posts", docId));
                    console.log("Tweet deleted:", docId);
                } catch (error) {
                    console.error("Error deleting tweet:", error);
                    alert("Failed to delete tweet.");
                }
            }
            postElement.remove();
        }
    });

    postElement.appendChild(contentParagraph);
    postElement.appendChild(tagsContainer);
    postElement.appendChild(timestampSpan);
    postElement.appendChild(deleteButton);

    // Make the tweet container clickable
    postElement.addEventListener("click", () => {
        viewTweetContent.textContent = tweetContent;
        viewTweetTags.innerHTML = "";
        tags.forEach((tag) => {
            const tagElement = document.createElement("span");
            tagElement.textContent = `#${tag.trim()}`;
            viewTweetTags.appendChild(tagElement);
        });
        viewTweetTimestamp.textContent = new Date(timestamp).toLocaleString();
        viewTweetPopup.style.display = "flex";
    });

    return postElement;
}

// Post a new tweet
postButton.addEventListener("click", async () => {
    const tweetContent = tweetInput.value.trim();
    const tags = tagsInput.value.split(",").map((tag) => tag.trim()).filter(Boolean);

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

        const newTweet = createTweetElement(tweetContent, tags, new Date(), docRef.id);
        tweetsContainer.prepend(newTweet);

        tweetInput.value = "";
        tagsInput.value = "";
        tweetPopup.style.display = "none";
    } catch (error) {
        console.error("Error posting tweet:", error);
        alert("Failed to post tweet. Please try again.");
    }
});

// Load all tweets
async function loadPosts() {
    tweetsContainer.innerHTML = "<p>Loading tweets...</p>";

    try {
        const postsQuery = query(collection(db, "posts"), orderBy("timestamp", "desc"));
        const querySnapshot = await getDocs(postsQuery);

        tweetsContainer.innerHTML = "";
        querySnapshot.forEach((doc) => {
            const post = doc.data();
            const timestamp = post.timestamp?.toDate ? post.timestamp.toDate() : new Date();  // Convert Firestore Timestamp to JS Date
        
            const postElement = createTweetElement(
                post.content,
                post.tags || [],
                timestamp,
                doc.id
            );
            tweetsContainer.appendChild(postElement);
        });
        
    } catch (error) {
        tweetsContainer.innerHTML = "<p>Failed to load tweets. Please try again.</p>";
        console.error("Error loading tweets:", error);
    }
}

// Load posts when the page loads
document.addEventListener("DOMContentLoaded", loadPosts);