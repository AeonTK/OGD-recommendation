$(document).ready(function(){
    $("#submit").on('click', function (event) {
        event.preventDefault();
        var email = 'huseynovramilism@gmail.com';
        var subject = 'OGD Survey';
        var emailBody = "";
        var count = 0;
        $("select").each(function() {
            if($(this).val()!=0){
                count++;
            }
            emailBody += $(this).val() + " "
        })
        if(count < 10){
            alert("You should rate at least 10 recommendations")
        }
        else{
            emailBody += $("#suggestions").val();
            window.location = 'mailto:' + email + '?subject=' + subject + '&body=' +   emailBody;
        }
    })
})