function SET100price(value) {
  var chkprict = document.getElementById("chkprict");
  var x = document.getElementById("choose-stock-price");
  var y = document.getElementById("choose-stock-pct");
  x.style.display = chkprict.checked ? "block" : "none";
}
function SET100pct(value) {
  var x = document.getElementById("choose-stock-price");
  var y = document.getElementById("choose-stock-pct");
  if (value === "Percent Change") {
    y.style.display = "none";
    x.style.display = "block";
  }
  else {}
}


function showDJIA() {
  var x = document.getElementById("choose-set");
  var y = document.getElementById("choose-djia");
  if (y.style.display === "none") {
    y.style.display = "block";
    x.style.display = "none";
  }
  else {}
}

function chooseSET() {
  // show selected stock name [SET100]
  selectElement = document.querySelector('#choose-stock-set');
  output = selectElement.options[selectElement.selectedIndex].text;
  document.querySelector('#stock-name-set').textContent = output;
}

function chooseDJIA() {
  // show selected stock name [SET100]
  selectElement = document.querySelector('#choose-stock-djia');
  output = selectElement.options[selectElement.selectedIndex].text;
  document.querySelector('#stock-name-djia').textContent = output;
}


