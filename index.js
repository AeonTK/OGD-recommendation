$(document).ready(function(){
    $("#submit").on('click', function (event) {
        event.preventDefault();
        var email = 'huseynovramilism@gmail.com';
        var subject = 'OGD Survey';
        var emailBody = "";
        $("select").each(function() {
            emailBody += $(this).val() + " "
        })
        window.location = 'mailto:' + email + '?subject=' + subject + '&body=' +   emailBody;
    })
})