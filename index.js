$(document).ready(function(){
    $("#submit").on('click', function (event) {
        event.preventDefault();
        var email = 'test@theearth.com';
        var subject = 'OGD Survey';
        var emailBody = 'Some blah';
        window.location = 'mailto:' + email + '?subject=' + subject + '&body=' +   emailBody;
    })
})