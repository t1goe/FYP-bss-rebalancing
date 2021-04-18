function hider(){
    var hidden = document.getElementById("hidden_fields");
    
    if (document.getElementById('simple').checked) {
        hidden.style.display = "none";
    } else {
        hidden.style.display = "block";
    }
}