
// Building data
const buildings = {
  cit: {
    name: "Center for Information Technology",
    description: "The Center for Information Technology (CIT) houses computer labs and classrooms designed for computer science and IT education. The building features modern computer equipment, smart classrooms, and collaborative spaces for students to work on projects and assignments.",
    floors: 5
  },
  tao: {
    name: "Technological Arts Office",
    description: "The Technological Arts Office (TAO) is focused on creative digital arts and design. This building contains design studios, multimedia labs, and exhibition spaces for students to showcase their work. It serves as a hub for interdisciplinary collaboration between technology and creative arts.",
    floors: 2
  },
  steerhub: {
    name: "Science, Technology, Engineering, and Environment Hub",
    description: "A hub for interdisciplinary research and collaboration focusing on sustainable technologies and environmental science. STEEHUB houses research laboratories, seminar rooms, and collaborative workspaces designed to foster innovation across different scientific disciplines.",
    floors: 5
  },
  fdc: {
    name: "Faculty Development Center",
    description: "A center dedicated to faculty research and development. The FDC provides resources, training facilities, and meeting spaces for faculty members to enhance their teaching methodologies and research capabilities. It includes conference rooms, training labs, and a resource library.",
    floors: 2
  },
  cics: {
    name: "College of Information and Computer Sciences",
    description: "The CICS building houses the College of Information and Computer Sciences with specialized labs for computing education and research. The building features programming labs, networking facilities, hardware workshops, and research spaces for advanced computing projects and artificial intelligence research.",
    floors: 5
  },
  coe: {
    name: "College of Engineering",
    description: "The College of Engineering building contains engineering labs and classrooms for various engineering disciplines. Facilities include materials testing labs, fluid dynamics labs, electronics workshops, and design studios where students can work on practical engineering projects and research.",
    floors: 3
  },
  ceafa: {
    name: "College of Engineering and Architecture Faculty Association",
    description: "Faculty offices and meeting rooms for engineering and architecture departments. CEAFA provides spaces for faculty collaboration, student consultations, and administrative functions for the engineering and architecture programs. It also houses small exhibition spaces for student design projects.",
    floors: 5
  }
};

// Get DOM elements
const modal = document.getElementById("buildingModal");
const buildingTitle = document.getElementById("buildingTitle");
const buildingDescription = document.getElementById("buildingDescription");
const floorTabs = document.getElementById("floorTabs");
const floorContentContainer = document.getElementById("floorContentContainer");
const closeBtn = document.querySelector(".close-btn");

// Add click event to all buildings
document.querySelectorAll(".building").forEach(building => {
  building.addEventListener("click", function() {
    // Remove active class from all buildings
    document.querySelectorAll(".building").forEach(b => {
      b.classList.remove("active");
    });

    // Add active class to clicked building
    this.classList.add("active");

    const buildingId = this.id;
    showBuildingDetails(buildingId);
  });
});

// Close the modal when clicking the X
closeBtn.addEventListener("click", function() {
  modal.style.display = "none";

  // Remove active class from all buildings
  document.querySelectorAll(".building").forEach(b => {
    b.classList.remove("active");
  });
});

// Close the modal when clicking outside the content
window.addEventListener("click", function(event) {
  if (event.target === modal) {
    modal.style.display = "none";

    // Remove active class from all buildings
    document.querySelectorAll(".building").forEach(b => {
      b.classList.remove("active");
    });
  }
});

// Function to show building details
function showBuildingDetails(buildingId) {
  const building = buildings[buildingId];

  // Update modal content
  buildingTitle.textContent = building.name;
  buildingDescription.textContent = building.description;

  // Clear previous floor tabs and content
  floorTabs.innerHTML = "";
  floorContentContainer.innerHTML = "";

  // Create floor tabs and content
  for (let i = 1; i <= building.floors; i++) {
    // Create tab
    const tab = document.createElement("button");
    tab.className = "floor-tab";
    tab.textContent = `Floor ${i}`;
    if (i === 1) tab.classList.add("active");
    floorTabs.appendChild(tab);

    // Create floor content
    const floorContent = document.createElement("div");
    floorContent.className = "floor-content";
    if (i === 1) floorContent.classList.add("active");

    if (buildingId === "cics") {
      // Create CICS floor layout
      const floorLayout = createCICSFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "fdc") {
      // Create FDC floor layout
      const floorLayout = createFDCFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "steerhub") {
      // Create Steer Hub floor layout
      const floorLayout = createSteerHubFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "tao") {
      // Create TAO floor layout
      const floorLayout = createTAOFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "cit") {
      // Create CIT floor layout
      const floorLayout = createCITFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "coe") {
      // Create COE floor layout
      const floorLayout = createCOEFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else if (buildingId === "ceafa") {
      // Create CEAFA floor layout
      const floorLayout = createCEAFAFloorLayout(i);
      floorContent.appendChild(floorLayout);
    } 
    else {
      // Create generic floor layout for other buildings
      const floorLayout = createGenericFloorLayout(buildingId, i);
      floorContent.appendChild(floorLayout);
    }

    floorContentContainer.appendChild(floorContent);

    // Add click event to tab
    tab.addEventListener("click", function() {
      // Remove active class from all tabs and content
      document.querySelectorAll(".floor-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".floor-content").forEach(c => c.classList.remove("active"));

      // Add active class to clicked tab and corresponding content
      this.classList.add("active");
      floorContentContainer.children[i-1].classList.add("active");
    });
  }

  // Show the modal
  modal.style.display = "block";
}

// Function to create CIT floor layout
function createCITFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `CIT Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];
  
// Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}
  
  return layout;
}

// Function to create TAO floor layout
function createTAOFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `TAO Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}

  return layout;
}

// Function to create Steer Hub floor layout
function createSteerHubFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `Steer Hub Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}

  return layout;
}

// Function to create FDC floor layout
function createFDCFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `FDC Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}

  return layout;
}

// Function to create CICS floor layout
function createCICSFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `CICS Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}

  return layout;
}

// Function to create COE floor layout
function createCOEFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `CICS Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}

  return layout;
}


// Function to create CEAFA floor layout
function createCEAFAFloorLayout(floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `CEAFA Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "0";
  corridor.style.top = "50px";
  corridor.style.width = "100%";
  corridor.style.height = "60%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Stairs left
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "0";
  stairsLeft.style.top = "50px";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "60%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  // Stairs right
  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "0";
  stairsRight.style.top = "50px";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "60%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Rooms on top row - different for each floor
  const roomColors = [
    "#7cb342", // light green
    "#7cb342", // indigo
    "#7cb342", // purple
    "#7cb342", // teal
    "#7cb342", // orange
    "#7cb342"  // red
  ];

  const roomLabels = [
    `${floorNumber}01`,
    `${floorNumber}02`,
    `${floorNumber}03`,
    `${floorNumber}04`,
    `${floorNumber}05`,
    `${floorNumber}06`
  ];

  for (let i = 0; i < 6; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${15 + i * 11.67}%`;
    room.style.top = "50px";
    room.style.width = "11.67%";
    room.style.height = "60%";
    room.style.backgroundColor = roomColors[i];
    room.textContent = roomLabels[i];
    layout.appendChild(room);
  }

  // Elevator
  const elevator = document.createElement("div");
  elevator.className = "room elevator";
  elevator.style.left = "40%";
  elevator.style.top = "80%";
  elevator.style.width = "20%";
  elevator.style.height = "15%";
  elevator.textContent = "Elevator";
  layout.appendChild(elevator);

  // Trash cans
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  // Create a container for all trash cans to ensure they stay in a line
const trashCanContainer = document.createElement("div");
trashCanContainer.style.position = "absolute";
trashCanContainer.style.display = "flex";
trashCanContainer.style.flexDirection = "row";
trashCanContainer.style.justifyContent = "space-between";
trashCanContainer.style.width = "30%";
trashCanContainer.style.left = "35%";
trashCanContainer.style.top = "73%";
layout.appendChild(trashCanContainer);

// Position trash cans in a perfect horizontal line
for (let i = 0; i < trashCanTypes.length; i++) { 
  const trashCan = document.createElement("div"); 
  trashCan.className = `trash-can ${trashCanTypes[i].type}`; 
  
  // Remove individual positioning and let flex handle it
  trashCan.style.position = "relative";
  trashCan.style.top = "0";
  trashCan.style.left = "0";
 
  trashCan.textContent = trashCanTypes[i].label; 
  trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`; 
  trashCan.classList.add("tooltip"); 
   
  trashCanContainer.appendChild(trashCan); // Add to container instead of layout
}
  return layout;
}

// Function to create generic floor layout
function createGenericFloorLayout(buildingId, floorNumber) {
  const layout = document.createElement("div");
  layout.className = "floor-layout";
  layout.style.position = "relative";

  // Floor title
  const title = document.createElement("h3");
  title.textContent = `${buildings[buildingId].name} - Floor ${floorNumber}`;
  title.style.textAlign = "center";
  title.style.padding = "10px";
  title.style.margin = "0";
  title.style.backgroundColor = "#4d774e";
  title.style.color = "white";
  layout.appendChild(title);

  // Main corridor
  const corridor = document.createElement("div");
  corridor.style.position = "absolute";
  corridor.style.left = "10%";
  corridor.style.top = "60px";
  corridor.style.width = "80%";
  corridor.style.height = "20%";
  corridor.style.backgroundColor = "#f0f0f0";
  layout.appendChild(corridor);

  // Rooms layout
  const roomsPerFloor = 4 + floorNumber; // Different number of rooms per floor
  const roomWidth = 80 / roomsPerFloor; // Distribute rooms evenly

  for (let i = 0; i < roomsPerFloor; i++) {
    const room = document.createElement("div");
    room.className = "room";
    room.style.left = `${10 + i * roomWidth}%`;
    room.style.top = "120px";
    room.style.width = `${roomWidth - 2}%`;
    room.style.height = "40%";

    // Different colors for different buildings
    const hue = (buildingId.charCodeAt(0) * 20 + floorNumber * 30) % 360;
    room.style.backgroundColor = `hsl(${hue}, 70%, 60%)`;

    room.textContent = `${buildingId.toUpperCase()}${floorNumber}${i+1}`;
    layout.appendChild(room);
  }

  // Stairs and elevators
  const stairsLeft = document.createElement("div");
  stairsLeft.className = "room stairs";
  stairsLeft.style.left = "10%";
  stairsLeft.style.top = "70%";
  stairsLeft.style.width = "15%";
  stairsLeft.style.height = "15%";
  stairsLeft.textContent = "Stairs";
  layout.appendChild(stairsLeft);

  const elevatorCenter = document.createElement("div");
  elevatorCenter.className = "room elevator";
  elevatorCenter.style.left = "40%";
  elevatorCenter.style.top = "70%";
  elevatorCenter.style.width = "20%";
  elevatorCenter.style.height = "15%";
  elevatorCenter.textContent = "Elevator";
  layout.appendChild(elevatorCenter);

  const stairsRight = document.createElement("div");
  stairsRight.className = "room stairs";
  stairsRight.style.right = "10%";
  stairsRight.style.top = "70%";
  stairsRight.style.width = "15%";
  stairsRight.style.height = "15%";
  stairsRight.textContent = "Stairs";
  layout.appendChild(stairsRight);

  // Add trash cans in different positions based on floor
  const trashCanTypes = [
    { type: "recycling", label: "R" },
    { type: "hazardous", label: "H" },
    { type: "organic", label: "O" },
    { type: "general", label: "G" }
  ];

  for (let i = 0; i < 4; i++) {
    const trashCan = document.createElement("div");
    trashCan.className = `trash-can ${trashCanTypes[i].type}`;

    // Place trash cans in different locations based on floor
    const xPos = (20 + i * 20 + floorNumber * 5) % 70 + 10;
    const yPos = floorNumber % 2 === 0 ? 85 : 90;

    trashCan.style.left = `${xPos}%`;
    trashCan.style.top = `${yPos}%`;

    trashCan.textContent = trashCanTypes[i].label;
    trashCan.dataset.tooltip = `${trashCanTypes[i].type.charAt(0).toUpperCase() + trashCanTypes[i].type.slice(1)} Waste`;
    trashCan.classList.add("tooltip");

    layout.appendChild(trashCan);
  }

  return layout;
}

