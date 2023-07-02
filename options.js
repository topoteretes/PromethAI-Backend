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
  let selectedMealId = document.getElementById('mealOptions').value;
  let selectedMeal = mealChoices.find(meal => meal.mealId === selectedMealId);
  selectedMeal.chosen = true;
  fetch(backendUrl + '/saveMealChoice', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(selectedMeal),
  })
  .then(response => response.json())
  .then(data => {
    console.log('Meal choice saved: ', data);
  });
}

// Event listener for the save meal choice button
document.getElementById('saveMealChoice').addEventListener('click', saveMealChoice);

// Load meal choices when the options page is loaded
window.onload = loadMealChoices;
```