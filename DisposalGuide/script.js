const infoBox = document.getElementById("infoBox");

const texts = {
    default: "No waste category hovered.",
    general: `
    General waste includes everyday non-hazardous items that cannot be recycled or composted.
    <br><br><strong>To properly dispose of them:</strong><br>
    Place in black or general waste bins as they are usually sent to landfills or waste-to-energy plants.
    <br><br><strong>Important Note:</strong><br>
    Do not put general waste in recycling or compost bins.
  `,
    hazardous: `
    Hazardous waste contains substances that are harmful to human health or the environment. These require special handling and disposal.
    <br><br><strong>To properly dispose of them:</strong><br>
    Follow local government or environmental authority guidelines. Take to certified hazardous waste collection sites or e-waste centers.
    <br><br><strong>Important Note:</strong><br>
    Never dispose in general or recycling bins.
  `,
    recyclables: `
    Recyclables are items that can be processed and reused to make new products.
    <br><br><strong>To properly dispose of them:</strong><br>
    Rinse and clean items before placing them in recycling bins. Check local recycling guidelines.
    <br><br><strong>Important Note:</strong><br>
    Do not include food-contaminated items or non-recyclable plastics.
  `
};

document.getElementById("general").addEventListener("mouseenter", () => {
    infoBox.innerHTML = texts.general;
});
document.getElementById("hazardous").addEventListener("mouseenter", () => {
    infoBox.innerHTML = texts.hazardous;
});
document.getElementById("recyclables").addEventListener("mouseenter", () => {
    infoBox.innerHTML = texts.recyclables;
});

["general", "hazardous", "recyclables"].forEach(id => {
    document.getElementById(id).addEventListener("mouseleave", () => {
        infoBox.innerHTML = texts.default;
    });
});
