L.CountrySelect = {};

L.CountrySelect.countries = mtl_list

L.CountrySelect = L.Control.extend({
	options: {
		position: 'topleft',
		title: 'Selectionnez une variable',
		exclude: [],
		include: [],
		countries: L.CountrySelect.countries,
	},
	onAdd: function(map2) {
		this.div = L.DomUtil.create('div','leaflet-countryselect-container');
		this.select = L.DomUtil.create('select','leaflet-countryselect',this.div);
		var content = '';
		
		if (this.options.title.length > 0 ){
			content += '<option>'+this.options.title+'</option>';
		}
		
		var countries = (Array.isArray(this.options.include) && this.options.include.length > 0) ? this.options.include : this.options.countries;
		var countryKeys = Object.keys(countries).sort();
		for (i in countryKeys){
			if (this.options.exclude.indexOf(countryKeys[i]) == -1){
				content+='<option>'+countryKeys[i]+'</option>';
			}
		}
		this.select.innerHTML = content;
		this.select.onmousedown = L.DomEvent.stopPropagation;
		
		return this.div;
	},
	on: function(type,handler){
		if (type == 'change'){
			this.onChange = handler;
			L.DomEvent.addListener(this.select,'change',this._onChange,this);			
		} else if (type == 'click'){ //don't need this here probably, but for convenience?
			this.onClick = handler;
			L.DomEvent.addListener(this.select,'click',this.onClick,this);			
		} else {
			console.log('CountrySelect - cannot handle '+type+' events.')
		}
	},
	_onChange: function(e) {
		var selectedCountry = this.select.options[this.select.selectedIndex].value;
		e.feature = this.options.countries[selectedCountry];
		this.onChange(e);
	}
});

L.countrySelect = function(id,options){
	return new L.CountrySelect(id,options);
};




