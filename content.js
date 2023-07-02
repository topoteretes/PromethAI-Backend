```javascript
let mealChoices = [];

// Function to load meal choices from the backend
function loadMealChoices() {
    fetch(backendUrl + '/mealChoices')
        .then(response => response.json())
        .then(data => {
            mealChoices = data;
            let mealOptions = document.getElementById('mealOptions');
            mealOptions.innerHTML = '';
            mealChoices.forEach(meal => {
                let option = document.createElement('option');
                option.value = meal.mealId;
                option.text = meal.mealName;
                mealOptions.appendChild(option);
            });
        });
}

// Function to save the selected meal choice
function saveMealChoice() {
    let mealOptions = document.getElementById('mealOptions');
    let selectedMeal = mealOptions.options[mealOptions.selectedIndex].value;
    let mealChoice = mealChoices.find(meal => meal.mealId === selectedMeal);
    mealChoice.chosen = true;
    fetch(backendUrl + '/saveMealChoice', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(mealChoice),
    });
}

// Event listener for the save meal choice button
document.getElementById('saveMealChoice').addEventListener('click', saveMealChoice);

// Load meal choices when the page loads
window.onload = loadMealChoices;
```