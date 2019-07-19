// initialize the map


//
var control;
var style;

//Basemaps
var baseIm = L.esri.basemapLayer("Imagery")
var baseImLabels = L.esri.basemapLayer('ImageryLabels')
var baseLayer = L.layerGroup([baseIm]);
var baseLayer2 = L.esri.basemapLayer("Topographic","TerrainLabels")
var baseMaps = {
"World Imagery": baseLayer,
"Contour": baseLayer2,
};

// Initialize map with layers and basemaps
var map2 = L.map("map2",{layers: [baseLayer], center: [50.538, -72.790], zoom: 6});

//Add scale to map
var featureGroup = L.featureGroup().addTo(map2);
L.control.scale({position: 'bottomleft'}).addTo(map2);

// Add zoom controls to map
map2.zoomControl.setPosition('bottomleft');

//Add first layer group to map (flow accumulation)
var rivLayers = L.layerGroup([
L.geoJson(flow_acc)]);
var overlayMaps = {
    "Rivières au Fleuve": rivLayers
};

// Add basemaps and overlays to map
var layerControl = L.control.layers(baseMaps, overlayMaps, {
sortLayers: true,}).addTo(map2);

//Add mouse coordinates to map
L.control.coordinates({
    position:"bottomright",
    decimals:4,
    decimalSeperator:",",
    labelTemplateLat:"Latitude: {y}",
    labelTemplateLng:"Longitude: {x}"
}).addTo(map2);

// Add dropdown menu to map
var select = L.countrySelect();
select.addTo(map2);




// Add export object to map
var customControl =  L.Control.extend({
  options: {
    position: 'bottomleft'
  },
  // On add
  onAdd: function (map) {
    var zoomName = 'leaflet-control-filelayer leaflet-control-zoom';
    var barName = 'leaflet-bar';
    var partName = barName + '-part';
    var container = L.DomUtil.create('div', zoomName + ' ' + barName);
    var link = L.DomUtil.create('a', zoomName + '-in ' + partName, container);
    link.type="button";
    link.title="Exporter les bassins versants";
    link.href = '#';
    link.style.backgroundImage = "url(https://image.flaticon.com/icons/svg/154/154838.svg)";
    link.style.backgroundSize = "20px 20px";
    link.style.width = '30px';
    link.style.height = '30px';
    // On mouse over event function
    container.onmouseover = function(){
      container.style.backgroundColor = 'pink';
    }
    // On mouse out event function
    container.onmouseout = function(){
      container.style.backgroundColor = 'white';
    }
    // On click event function
    container.onclick = function(){
        var data = WaterItems.toGeoJSON();
        console.log(data.features)
        if (Array.isArray(data.features) && data.features.length) {
            // Stringify the GeoJson
            var convertedData = 'text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data));
            // Create export
            link.setAttribute('href', 'data:' + convertedData);
            link.setAttribute('download','data.geojson');
        }
    }
    return container;
  }
});
map2.addControl(new customControl());




// Create drawnItems feature Group and add to map
var drawnItems = new L.FeatureGroup();
map2.addLayer(drawnItems);
// Initialise the draw control and pass it the FeatureGroup of editable layers : drawnItems
var drawControl2 = new L.Control.Draw({
    draw: {
        position: 'topleft',
        circle: {
            shapeOptions: {
                color: '#662d91'
            }
        }
    },
    edit: {
        featureGroup: drawnItems
    }
});
map2.addControl(drawControl2);
// On Draw event function
map2.on(L.Draw.Event.CREATED, function (e) {
  var type = e.layerType
  var layer = e.layer;
  // Do whatever else you need to. (save to db, add to map etc)
  drawnItems.addLayer(layer);
  if (type === 'marker') {
    layer.bindPopup('' + layer.getLatLng()).openPopup();
    }
});




// Get all markers in the viewport
function getFeaturesInView() {
  var features = [];
  map2.eachLayer( function(layer) {
    if(layer instanceof L.Marker) {
      if(map2.getBounds().contains(layer.getLatLng())) {
        console.log(layer.getLatLng())
        var lat = layer.getLatLng().lat;
        var lng = layer.getLatLng().lng
        features.push([lat,lng]);
      }
    }
  });
  return features;
}

// Define information control object to map
var info;
var info = L.control({position: 'topright'});
    // On add
    info.onAdd = function(map2) {
      this._div = L.DomUtil.create("div", "info");
      this.update();
      return this._div;
    };
    // On update
   info.update = function(props) {
      this._div.innerHTML =
       "<h4>Variable physiographique</h4>" +
       (props
         ? "<b>" +
          props.NOM +
          "</b><br />" +
          Math.floor(props[property_col])
         : "Choisissez un bassin avec votre curseur");
    };
info.addTo(map2);

// Define Colors for style and legend
function getColor(d) {
  return d > 40000
   ? "#800026"
   : d > 30000
     ? "#BD0026"
     : d > 20000
      ? "#E31A1C"
      : d > 10000
        ? "#FC4E2A"
        : d > 5000
         ? "#FD8D3C"
         : d > 500 ? "#FEB24C" : d > 100 ? "#FED976" : "#FFEDA0";
}

// Add legend
var legend;
var legend = L.control({ position: "bottomright" });
// On add
legend.onAdd = function(map2) {
  var div = L.DomUtil.create("div", "info legend"),
   grades = [0, 100, 500, 5000, 10000, 20000, 30000, 40000],
   labels = [],
   from,
   to;
  for (var i = 0; i < grades.length; i++) {
   from = grades[i];
   to = grades[i + 1];
   labels.push(
     '<i style="background:' +
      getColor(from + 1) +
      '"></i> ' +
      from +
      (to ? "&ndash;" + to : "+")
   );
  }
  div.innerHTML = labels.join("<br>");
  return div;
};

var json_bassinsNaturels = function () {
    var jsonTemp = null;
    $.ajax({
        'async': false,
        'url': "../static/js/BassinsNaturels_Poly_Simplifie.json",
        'success': function (data) {
            jsonTemp = data;
        }
    });
    return jsonTemp;
}();
console.log(json_bassinsNaturels);

var json_bassinsExploites = function () {
    var jsonTemp = null;
    $.ajax({
        'async': false,
        'url': "../static/js/BassinsExploites_Poly_Simplifie.json",
        'success': function (data) {
            jsonTemp = data;
        }
    });
    return jsonTemp;
}();
console.log(json_bassinsExploites);

var json_bassinsHSAMI = function () {
    var jsonTemp = null;
    $.ajax({
        'async': false,
        'url': "../static/js/HSAMI_Poly_Simplifie.json",
        'success': function (data) {
            jsonTemp = data;
        }
    });
    return jsonTemp;
}();
console.log(json_bassinsHSAMI);


var geojson;
var geojson2;
var geojson3;

function style(feature) {
  return {
   weight: 2,
   opacity: 1,
   color: "white",
   dashArray: "3",
   fillOpacity: 0.7
  };
}

function resetHighlight(e) {
  var layer = e.target;
  layer.setStyle(style);
  info.update();
}
function zoomToFeature(e) {
//          map.fitBounds(e.target.getBounds());
}
var property_col;
// Define highlight function
function highlightFeature(e) {
  var layer = e.target;
//   layer.setStyle({
//       weight: 2,
//       opacity: 1,
//       color: "white",
//       dashArray: "3",
//       fillColor: "blue"
//   });
//  if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
//   layer.bringToFront();
//  }
  info.update(layer.feature.properties);
}

function onEachFeature(feature, layer) {
  layer.on({
   mouseover: highlightFeature,
   mouseout: resetHighlight,
   click: zoomToFeature
  });
}

// Define new geojson
geojson = L.geoJson(json_bassinsHSAMI, {
  onEachFeature: onEachFeature,
});

geojson2 = L.geoJson(json_bassinsExploites, {
  onEachFeature: onEachFeature,
});

geojson3 = L.geoJson(json_bassinsNaturels, {
  onEachFeature: onEachFeature,
});


var layerGroup1 = new L.LayerGroup([
    geojson]);

layerControl.addOverlay(layerGroup1, "HSAMI (bassins)");

var layerGroup2 = new L.LayerGroup([
    geojson2]);

layerControl.addOverlay(layerGroup2, "Bassins Exploités");

var layerGroup3 = new L.LayerGroup([
    geojson3]);

layerControl.addOverlay(layerGroup3, "Bassins Naturels");
// select : on change

// Create WaterItems feature Group and add to map
var WaterItems = new L.FeatureGroup();
map2.addLayer(WaterItems);

// Initialise the draw control and pass it the FeatureGroup of editable layers : WaterItems
var drawControl = new L.Control.Draw({
  position: 'bottomleft',
  draw: false,
  edit: {
    featureGroup: WaterItems
  }
});
map2.addControl(drawControl);

// Add file loading/drag and drop object
L.Control.FileLayerLoad.LABEL = '<img class="icon" src="../static/img/folder.svg" alt="file icon"/>';
// Configure the button and object interactions
control = L.Control.fileLayerLoad({
position: 'bottomleft',
// Allows you to use a customized version of L.geoJson.
// For example if you are using the Proj4Leaflet leaflet plugin,
// you can pass L.Proj.geoJson and load the files into the
// L.Proj.GeoJson instead of the L.geoJson.
layer: L.geoJson,
// See http://leafletjs.com/reference.html#geojson-options
layerOptions: {style: style},
// Add to map after loading (default: true) ?
addToMap: true,
// File size limit in kb (default: 1024) ?
fileSizeLimit: 15000,
// Restrict accepted file formats (default: .geojson, .json, .kml, and .gpx) ?
formats: [
    '.geojson',
    '.kml']
});
// Add control to map
control.addTo(map2);

// On load function
control.loader.on('data:loaded', function (e) {
    var layer = e.layer;
    console.log(layer);
    layer.eachLayer(
        function(l){
            WaterItems.addLayer(l);
    });
    layerControl.addOverlay(layer, e.filename);
});

// Add watershed delimitation object to map
var customControl2 =  L.Control.extend({
  options: {
    position: 'bottomleft'
  },
    // On add
  onAdd: function (map) {
    var zoomName = 'leaflet-control-filelayer leaflet-control-zoom';
    var barName = 'leaflet-bar';
    var partName = barName + '-part';
    var container = L.DomUtil.create('div', zoomName + ' ' + barName);
    var link = L.DomUtil.create('a', zoomName + '-in ' + partName, container);
    link.type="button";
    link.title="Delimiter les bassins versants";
    link.href = '#';
    link.style.backgroundImage = "url(https://cdn4.iconfinder.com/data/icons/green-energy-ecology-colours/91/Green_Energy_-_Ecology_12-512.png)";
    link.style.backgroundSize = "20px 20px";
    link.style.width = '30px';
    link.style.height = '30px';
    // on mouse over event function
    container.onmouseover = function(){
      container.style.backgroundColor = 'pink';
    }
    // on mouse out event function
    container.onmouseout = function(){
      container.style.backgroundColor = 'white';
    }
    // on click event function
    container.onclick = function(){
      var c=getFeaturesInView();
      console.log(typeof(c))
      console.log(c)
      $.getJSON($SCRIPT_ROOT + '/_calc_watershed', {
        a: JSON.stringify(c),
      }, function(data) {
            console.log(data.features.properties)
            console.log(this)
            var imageOverlayNew = L.geoJSON(data).addTo(map2);
            let i = 0;
            map2.eachLayer(function(){ i += 1; });
            console.log('Map has', i, 'layers.');
            // Remove previous WaterItems in layer control
            layerControl.removeLayer(WaterItems);
            // Add newly created watershed polygon to WaterItems feature group
            imageOverlayNew.eachLayer(
                function(l){
                    WaterItems.addLayer(l);
            });

            WaterItems.eachLayer(function(e){
            console.log(e.feature);
                e.on('mouseover', function (event) {
                  var layer = event.target;
                  info.update(layer.feature.properties);
                  }
                );
                e.on('mouseout', function (event) {
                  info.update();
                  }
                );
                e.on('click', function (event) {
                    var layer = event.target;

                    layer.feature.type = layer.feature.type || "Feature"; // Intialize feature.type
                    var props = layer.feature.properties = layer.feature.properties || {}; // Intialize feature.properties
                    var template = '<form id="popup-form"><label for="input-speed">NOM : </label><input id="input-speed" class="popup-input" type="string" /><button id="button-submit" type="button">Sauvegarder</button></form>';
                    layer.bindPopup(template);
                    layer.openPopup();

                    var inputSpeed = L.DomUtil.get('input-speed');
                    inputSpeed.value = props.NOM || "";
                    L.DomEvent.addListener(inputSpeed, 'change', function (e) {
                    props.NOM = e.target.value;
                    });

                    var buttonSubmit = L.DomUtil.get('button-submit');
                    L.DomEvent.addListener(buttonSubmit, 'click', function (e) {
                    layer.closePopup();
                    });
                  });
                e.setStyle({
                   weight: 2,
                   opacity: 1,
                   color: "white",
                   dashArray: "3",
                   fillOpacity: 0.7,
                   fillColor: getColor(e.feature.properties[property_col])
                  }
                );
            });
            // Delete features such as markers
            drawnItems.clearLayers();
            console.log(WaterItems)
            // Add WaterItems group to the layer control
            layerControl.addOverlay(WaterItems, "Délimitation de bassins");
        }
      );
    }
    return container;
  }
});
map2.addControl(new customControl2());


// Here we are creating control to show it on the map;
L.control.browserPrint({
    printModes: [
        L.control.browserPrint.mode.auto("Expoorter en PNG"),
        L.control.browserPrint.mode.landscape(),
        L.control.browserPrint.mode.portrait(),
        L.control.browserPrint.mode.auto(),
        L.control.browserPrint.mode.custom()
    ]}).addTo(map2);
map2.on("browser-print-start", function(e){

    /*on print start we already have a print map and we can create new control and add it to the print map to be able to print custom information */
    legend.addTo(e.printMap);
});



change_val = 1;
select.on('change', function(e){
    // define variables
    var values = [];
    //Do nothing if undefined
    if (e.feature === undefined){
        return;
    }
    // Replace previous geojson with current for next iteration
    property_col = this.select.options[this.select.selectedIndex].value
    //Info Update;
    info.update = function(props) {
      this._div.innerHTML =
       "<h4>Variable physiographique</h4>" +
       (props
         ? "<b>" +
          props.NOM +
          "</b><br />" +
          Math.floor(props[property_col])
         : "Choisissez un bassin avec votre curseur");
    };

    // Get all values
    geojson.eachLayer(function(e) {
    var value = Number(e.feature.properties[property_col]);
    values.push(value);
    });
    facteur = Math.ceil(Math.max.apply(Math, values)/500)
    facteur==1 ? facteur=0.5:facteur ==1
    var breaks = ss.quantile(values, [0.05, .15, 0.3, 0.45, 0.65, 0.8, 1]);

    // Assign legend color
    var multiplicateur=[];
    for (i = 0; i < 7; i++) {
      multiplicateur.push(Math.floor(breaks[i] / facteur) * facteur);
    }
    function getColor(d) {
      return d > multiplicateur[6]
       ? "#800026"
       : d > multiplicateur[5]
         ? "#BD0026"
         : d > multiplicateur[4]
          ? "#E31A1C"
          : d > multiplicateur[3]
            ? "#FC4E2A"
            : d > multiplicateur[2]
             ? "#FD8D3C"
             : d > multiplicateur[1] ? "#FEB24C"
              : d > multiplicateur[0] ? "#FED976" : "#FFEDA0";
    }
    //Update style function
    function style(feature) {
      return {
       weight: 2,
       opacity: 1,
       color: "white",
       dashArray: "3",
       fillOpacity: 0.7,
       fillColor: getColor(feature.properties[property_col])
      };
    }
    function highlightFeature(e) {
      var layer = e.target;
//      if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
//       layer.bringToFront();
//      }
      info.update(layer.feature.properties);
    }
    function resetHighlight(e) {
      var layer = e.target;
//      layer.setStyle(style);
      info.update();
    }
    //Apply style to geojson
    geojson.setStyle(style);
    geojson2.setStyle(style);
    geojson3.setStyle(style);


    if (change_val== 1){
        WaterItems.eachLayer(function(e){
        // console.log(e.feature.properties);
            e.on('mouseover', function (event) {
              var layer = event.target;
              info.update(layer.feature.properties);
              }
            );
            e.on('mouseout', function (event) {
              info.update();
              }
            );
            e.on('click', function (event) {
                var layer = event.target;

                layer.feature.type = layer.feature.type || "Feature"; // Intialize feature.type
                var props = layer.feature.properties = layer.feature.properties || {}; // Intialize feature.properties
                var template = '<form id="popup-form"><label for="input-speed">NOM : </label><input id="input-speed" class="popup-input" type="string" /><button id="button-submit" type="button">Sauvegarder</button></form>';
                layer.bindPopup(template);
                layer.openPopup();

                  var inputSpeed = L.DomUtil.get('input-speed');
                  inputSpeed.value = props.NOM || "";
                  L.DomEvent.addListener(inputSpeed, 'change', function (e) {
                    props.NOM = e.target.value;
                  });

                  var buttonSubmit = L.DomUtil.get('button-submit');
                  L.DomEvent.addListener(buttonSubmit, 'click', function (e) {
                    layer.closePopup();
                  });
//                var inputSpeed = L.DomUtil.get('input-speed');
//                var buttonSubmit = L.DomUtil.get('button-submit');
//                L.DomEvent.addListener(buttonSubmit, 'click', function (e) {
//                  layer.closePopup();
//                });
              }
            );
            e.setStyle({
               weight: 2,
               opacity: 1,
               color: "white",
               dashArray: "3",
               fillOpacity: 0.7,
               fillColor: getColor(e.feature.properties[property_col])
              }
            );
        });
        change_val = 1;
    }
    // Update legend
    legend.onAdd = function(map2) {
      var div = L.DomUtil.create("div", "info legend"),
       grades = breaks,
       labels = [],
       from,
       to;
      for (var i = 0; i < grades.length; i++) {
       from = Math.floor(grades[i] / facteur) * facteur;
       to = Math.floor(grades[i + 1] / facteur) * facteur;
       labels.push(
         '<i style="background:' +
          getColor(from + 1) +
          '!important;"></i> ' +
          from +
          (to ? "&ndash;" + to : "+")
       );
      }
      div.innerHTML = labels.join("<br>");
      return div;
    };
    legend.addTo(map2);

});
