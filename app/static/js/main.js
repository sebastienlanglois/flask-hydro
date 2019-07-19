$(document).ready(function () {
    $('#sidebarCollapse').on('click', function () {
        $('#sidebar').toggleClass('active');
    });
});

$('#sidebarCollapse').click();

$(function() {
$('a#bassin1p').bind('click', function() {
  var c=getFeaturesInView();
  console.log(typeof(c))
  console.log(c)
  $.getJSON($SCRIPT_ROOT + '/_calc_watershed', {
    a: JSON.stringify(c),
  }, function(data) {
        var imageOverlayNew = L.geoJSON(data).addTo(map2);
        let i = 0;
        map2.eachLayer(function(){ i += 1; });
        console.log('Map has', i, 'layers.');
        layerControl.removeLayer(WaterItems);
//        WaterItems.addLayer(imageOverlayNew);
        imageOverlayNew.eachLayer(
            function(l){
                WaterItems.addLayer(l);
        });

        drawnItems.clearLayers();
        console.log(WaterItems)
        layerControl.addOverlay(WaterItems, "Délimitation de bassins");
    }
  );
  return false;
});
});


//document.getElementById('export').onclick = function(e) {
//    // Extract GeoJson from featureGroup
//    var data = WaterItems.toGeoJSON();
//
//    // Stringify the GeoJson
//    var convertedData = 'text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data));
//
//    // Create export
//    document.getElementById('export').setAttribute('href', 'data:' + convertedData);
//    document.getElementById('export').setAttribute('download','data.geojson');
//}
//
//
//window.onload = function(){
//    document.getElementById('delete').onclick = function(e) {
//        console.log("TEst")
//        drawnItems.clearLayers();
//    }
//};
//
