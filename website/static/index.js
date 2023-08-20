// function editDataset()

const dropContainer = document.getElementById("dropcontainer")
const fileInput = document.getElementById("file")

dropContainer.addEventListener("dragover", (e) => {
// prevent default to allow drop
e.preventDefault();
}, false);

dropContainer.addEventListener("dragenter", () => {
dropContainer.classList.add("drag-active");
});

dropContainer.addEventListener("dragleave", () => {
dropContainer.classList.remove("drag-active");
});

dropContainer.addEventListener("drop", (e) => {
e.preventDefault();
dropContainer.classList.remove("drag-active");
fileInput.files = e.dataTransfer.files;
// Submit the form
const form = dropContainer.closest("form");
form.submit();
});

function deleteDataset(datasetId){
    fetch("/delete-dataset",{
        method: "POST",
        body: JSON.stringify({datasetId:datasetId})
    }).then((_res) => {
        window.location.href = "/profile";
    });
}

function deleteFeature(featureId){
    fetch("/delete-feature",{
        method: "POST",
        body: JSON.stringify({featureId:featureId})
    }).then((_res) => {
        window.location.href = "/profile";
    });
}

function editDataset(){
    alert("This function is still to be added")
}

function submitForm() {
    const form = document.querySelector("form"); // Change this selector to match your form's actual structure
    form.submit();
}