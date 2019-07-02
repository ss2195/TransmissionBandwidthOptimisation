//jQuery and javascript file for some front end magic

$(document).ready(function() {

    function change_radio(opt){
        var value = $( 'input[name='+opt+']:checked' ).val();
        return value
    }


    $(document).on('click', 'tr' ,function(){
        var trChild = $(this).children();
        console.log(trChild);
        var selected = $(this).hasClass("selected");
        $("#table tr").removeClass("selected");
        if(!selected)
            $(this).addClass("selected");
        var sitepop = [trChild[1].innerText,trChild[2].innerText];
        console.log(sitepop);
        $.ajax({
            type: 'GET',
            url: '/site_data',
            data:{ 'sitepop': sitepop
                    },
            dataType:"text",
            success:function(demo){
                console.log("successful")
            }

    });
    });


/*
    $("input[name$='opt1']").click(function() {
        var test = $(this).val();
        $("div.threshold").hide();
        $("#"+test).show();
    });

   */
    $("input[name$='opt1']").click(function()  {
    if($('#0')[0].checked)
        $('#message').text("Set Health Threshold (0.00-1.00) : ");
    if($('#1')[0].checked)
        $('#message').text("Set Health Threshold (0.00-1.00) : ");
    if($('#2')[0].checked)
        $('#message').text("Set per site bandwidth in(kB): ");
    if($('#3')[0].checked)
        $('#message').text("Set per site bandwidth in(kB): ");
    if($('#4')[0].checked)
        $('#message').text("Set occupancy threshold: ");
    });



    $('#load').click(function() {
    if($.trim($('#threshold').val()) == '')
        alert('Input can not be left blank');
    else{

        var inp = $("input[name$='threshold']").val();
        var mode = change_radio('opt1')
        console.log(mode)
        console.log(inp)
        $.ajax({
            type: 'GET',
            url: '/display_table',
            data:{ 'thold':inp,
                    'mode':mode
                    },
            dataType:"text",
            success:function(csvdata)
            {
                var employee_data=csvdata.split(/\r?\n|\r/);
                var table_data='<table class="table table-bordered table-striped" id="table">';
                for(var count=0;count<employee_data.length-2; count++)
                {
                    var cell_data=employee_data[count].split(",");
                    table_data+='<tr>';
                    for(var cell_count=0;cell_count<cell_data.length; cell_count++)
                    {
                        if (count==0)
                        {
                            table_data+='<th>'+cell_data[cell_count]+'</th>';
                        }
                        else
                        {
                            table_data +='<td>'+cell_data[cell_count]+'</td>';
                        }
                    }
                    table_data += '</tr>';
                }
                table_data += '</table>';
                $('#employee_table').html(table_data);
            }
        });

       }
    });

$('#graph').click(function(){

    var trChild = $('.selected').children();
    var sitepop1 = ['',''];
    var tddfdd = change_radio('opt2')
    console.log(tddfdd)
    try{
            sitepop1 = [trChild[1].innerText,trChild[2].innerText];
    }
    catch{
    }
    if(sitepop1[0]=='')
        alert("Select a site");
    else if(sitepop1[0] == "Site ID")
        alert("Select a valid site");
    else{
        console.log(sitepop1);
        $.ajax({
            type: 'GET',
            url: '/site_pop_data',
            data:{ 'sitepop': sitepop1,
                    'tddfdd' : tddfdd
                    },
            dataType:"text",
            success:function(demo){
                $('#myModalHorizontal').modal('show');
                console.log("successful")
            }

            });

            }
    });
    /*
    $(document).on('click','#getgraph',function(){
    //e.preventDefault();
    $('#myModalHorizontal').modal('hide');
        var opt3 = change_radio('opt3')
        console.log(opt3)
        $.ajax({
            type: 'GET',
            url:'/getGraph',
            data: {
                'opt3': opt3
            },
            success: function(file){
                //console.log(file);
                //var grph = window.open("","graph","width=400,height=400");
                //console.log(file)
                //grph.document.write(file);
                //$('html').load(file)
                console.log(file)
            }

        });
        });
        */
/*
     $(document).on('click', '#map', function(){
        $.ajax({
            type: 'GET',
            url: '/getMap',
            success: function(mapData){
                $("html").empty();
                $("html").append(mapData);
            }

        });

     });
     */
     $('#map').click(function(event){
    // $('#form1').on('submit', function( event ){
     var trChild = $('.selected').children();
     console.log
     var sitepop1 = ['',''];
     sitepop1 = [trChild[1].outerText,trChild[2].outerText];
     console.log(sitepop1);
     if(sitepop1[0]==''){
        event.preventDefault();
        alert("Select a site");
     }
     if(sitepop1[0] == "Site ID"){
        event.preventDefault();
        alert("Select a valid site");
     }

     });

    });




function download_csv(csv, filename) {

    var csvFile;
    var downloadLink;

    // CSV FILE
    csvFile = new Blob([csv], {type: "text/csv"});

    // Download link
    downloadLink = document.createElement("a");

    // File name
    downloadLink.download = filename;

    // We have to create a link to the file
    downloadLink.href = window.URL.createObjectURL(csvFile);

    // Make sure that the link is not displayed
    downloadLink.style.display = "none";

    // Add the link to your DOM
    document.body.appendChild(downloadLink);

    // Lanzamos
    downloadLink.click();
}

function export_table_to_csv(html, filename) {
	var csv = [];
	var rows = document.querySelectorAll("table tr");

    for (var i = 0; i < rows.length; i++) {
		var row = [], cols = rows[i].querySelectorAll("td, th");

        for (var j = 0; j < cols.length; j++)
            row.push(cols[j].innerText);

		csv.push(row.join(","));
	}

    // Download CSV
    download_csv(csv.join("\n"), filename);
}

document.querySelector("#download").addEventListener("click", function () {

    var html = document.querySelector("table").outerHTML;
	export_table_to_csv(html, "table.csv");

});

