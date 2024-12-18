$(document).ready(function () {

    $(document).ready(function () {
        $(".description").each(function (i) {
            var len = $(this).html().length;
            if (len > 400) {
                $(this).html($(this).html().substr(0, 400) + '...');
                $(this).append($('<a href="'+$(this).data("more")+'" target="_blank">Read more</a>'));
            }
        });
    });

    $("#submit").on('click', function (event) {
        event.preventDefault();
        var email = 'ramil.huseynov@ut.ee';
        var subject = 'OGD Survey';
        var emailBody = "Ratings: ";
        var count = 0;
        $(".rating").each(function () {
            if ($(this).val() != 0) {
                count++;
            }
            emailBody += $(this).val() + " "
        })
        if (count < 10) {
            alert("You should rate at least 10 recommendations")
        }
        else {
            emailBody += "\n Suggestions: " + $("#suggestions").val();
            emailBody += "\n Portal frequency: " + $("#portal-freq").val();
            emailBody += "\n EU portal frequency: " + $('#eu-portal-freq').val();
            emailBody += "\n Navigation easiness: " + $('#easy-navigation').val();
            emailBody += "\n Expectation: " + $('#expectation').val();
            window.location = 'mailto:' + email + '?subject=' + subject + '&body=' + emailBody;
        }
    })
})