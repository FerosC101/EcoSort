const infoBox = document.getElementById('infoBox');

const wasteInfo = {
    general: `
    <strong>General Waste:</strong><br>
    General waste includes everyday non-hazardous items that cannot be recycled or composted. These are typical household or office waste materials.<br><br>
    <strong>To properly dispose of them:</strong><br>
    Place in black or general waste bins as they are usually sent to landfills or waste-to-energy plants.<br><br>
    <strong>Important Note:</strong><br>
    Do not put general waste in recycling or compost bins.
  `,
    hazardous: `
    <strong>Hazardous Waste:</strong><br>
    Hazardous waste contains substances that are harmful to human health or the environment. These require special handling and disposal.<br><br>
    <strong>To properly dispose of them:</strong><br>
    Follow local government or environmental authority guidelines for specific disposal instructions. If possible, take them to a certified hazardous waste collection site or municipal collection event.<br>
    Use e-waste recycling centers for electronics.<br><br>
    <strong>Important Note:</strong><br>
    Never dispose in general or recycling bins.
  `,
    recyclable: `
    <strong>Recyclables:</strong><br>
    Recyclables are items that can be processed and reused to make new products.<br><br>
    <strong>To properly dispose of them:</strong><br>
    Rinse and clean items before placing them in recycling bins. Consider checking local recycling guidelines for accepted materials.<br><br>
    <strong>Important Note:</strong><br>
    Do not include food-contaminated items or non-recyclable plastics.
  `,
    residual: `
    <strong>Residual Waste:</strong><br>
    Residual waste refers to non-recyclable and non-hazardous waste that remains after all recyclable and reusable materials have been separated. They are collected by municipal services and usually landfilled or incinerated.<br><br>
    <strong>To properly dispose of them:</strong><br>
    Place in residual waste bins (often gray or black, depending on the locality).<br><br>
    <strong>Important Note:</strong><br>
    Do not mix with recyclables or compostables.
  `
};
function showInfo(type) {
    infoBox.innerHTML = wasteInfo[type];
    infoBox.classList.add("active");
}

function resetInfo() {
    infoBox.innerHTML = "<div class='info-content'><p>No waste category hovered.</p></div>";
    infoBox.classList.remove("active");
}

document.getElementById('general').addEventListener('mouseenter', () => showInfo('general'));
document.getElementById('hazardous').addEventListener('mouseenter', () => showInfo('hazardous'));
document.getElementById('recyclables').addEventListener('mouseenter', () => showInfo('recyclable'));
document.getElementById('residual').addEventListener('mouseenter', () => showInfo('residual'));

document.querySelectorAll('.bin').forEach(bin => {
    bin.addEventListener('mouseleave', resetInfo);
});


