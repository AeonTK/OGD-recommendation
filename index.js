$(document).ready(function () {

    $(document).ready(function () {
        $(".description").each(function (i) {
            var len = $(this).html().length;
            if (len > 500) {
                $(this).html($(this).html().substr(0, 500) + '...');
                $(this).append($('<a href="'+$(this).data("more")+'" target="_blank">Read more</a>'));
            }
        });
    });

    $("#submit").on('click', function (event) {
        event.preventDefault();
        var email = 'huseynovramilism@gmail.com';
        var subject = 'OGD Survey';
        var emailBody = "";
        var count = 0;
        $("select").each(function () {
            if ($(this).val() != 0) {
                count++;
            }
            emailBody += $(this).val() + " "
        })
        if (count < 10) {
            alert("You should rate at least 10 recommendations")
        }
        else {
            emailBody += $("#suggestions").val();
            window.location = 'mailto:' + email + '?subject=' + subject + '&body=' + emailBody;
        }
    })
})