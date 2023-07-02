```javascript
let backendUrl = "https://example.com/api";
let mealChoices = [];

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ mealChoices: [] });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.message === "SAVE_COMMENT") {
    saveComment(request.data);
  } else if (request.message === "LOAD_COMMENTS") {
    loadComments(sendResponse);
    return true;
  } else if (request.message === "SAVE_MEAL_CHOICE") {
    saveMealChoice(request.data);
  } else if (request.message === "LOAD_MEAL_CHOICES") {
    loadMealChoices(sendResponse);
    return true;
  }
});

function connectToBackend() {
  // Implement connection to backend here
}

function saveComment(comment) {
  // Implement saving comment to backend here
}

function loadComments(callback) {
  // Implement loading comments from backend here
}

function saveMealChoice(mealChoice) {
  chrome.storage.sync.get("mealChoices", (data) => {
    mealChoices = data.mealChoices;
    mealChoices.push(mealChoice);
    chrome.storage.sync.set({ mealChoices: mealChoices });
  });
}

function loadMealChoices(callback) {
  chrome.storage.sync.get("mealChoices", (data) => {
    callback(data.mealChoices);
  });
}
```