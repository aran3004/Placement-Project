// function editDataset()

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