document.addEventListener("DOMContentLoaded", function () {
    const openFormBtn = document.getElementById("openForm");
    const closeFormBtn = document.getElementById("closeForm");
    const popupForm = document.getElementById("popupForm");

    openFormBtn.addEventListener("click", function () {
        popupForm.style.display = "flex";
    });

    closeFormBtn.addEventListener("click", function () {
        popupForm.style.display = "none";
    });

    popupForm.addEventListener("click", function (event) {
        if (event.target === popupForm) {
            popupForm.style.display = "none";
        }
    });
});
