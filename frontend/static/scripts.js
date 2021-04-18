function hider(){
    var hidden = document.getElementById("hidden_fields");

    if (document.getElementById('trigger').checked) {
        hidden.style.display = "none";
    } else {
        hidden.style.display = "block";
    }
}