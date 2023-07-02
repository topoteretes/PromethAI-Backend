```javascript
let backendUrl;
let mealChoices = [];

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submitComment').addEventListener('click', saveComment);
    document.getElementById('saveMealChoice').addEventListener('click', saveMealChoice);
    loadMealChoices();
});

function connectToBackend() {
    // Implement connection to backend
}

function saveComment() {
    let commentBox = document.getElementById('commentBox');
    let commentText = commentBox.value;
    let commentId = new Date().getTime();
    let webPageUrl = window.location.href;
    let timestamp = new Date();

    let userComment = {
        commentId: commentId,
        commentText: commentText,
        webPageUrl: webPageUrl,
        timestamp: timestamp
    };

    chrome.runtime.sendMessage({message: 'SAVE_COMMENT', payload: userComment}, function(response) {
        console.log(response);
    });

    commentBox.value = '';
}

function loadComments() {
    chrome.runtime.sendMessage({message: 'LOAD_COMMENTS'}, function(response) {
        console.log(response);
    });
}

function saveMealChoice() {
    let mealOptions = document.getElementById('mealOptions');
    let chosenMeal = mealOptions.value;
    let mealId = new Date().getTime();
    let chosen = true;

    let mealChoice = {
        mealId: mealId,
        mealName: chosenMeal,
        chosen: chosen
    };

    chrome.runtime.sendMessage({message: 'SAVE_MEAL_CHOICE', payload: mealChoice}, function(response) {
        console.log(response);
    });

    mealOptions.value = '';
}

function loadMealChoices() {
    chrome.runtime.sendMessage({message: 'LOAD_MEAL_CHOICES'}, function(response) {
        mealChoices = response;
        let mealOptions = document.getElementById('mealOptions');
        mealOptions.innerHTML = '';
        mealChoices.forEach(function(meal) {
            let option = document.createElement('option');
            option.text = meal.mealName;
            option.value = meal.mealName;
            mealOptions.add(option);
        });
    });
}
```