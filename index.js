$(document).ready(function () {
    Parse.initialize("mngedHY5lYolVxO2JUasuSk6K3VStvqxQvv1v1p7", "l2NHRBsGFqZzzh23E5YAAHIFOy2WCZG8fk76iwir"); //PASTE HERE YOUR Back4App APPLICATION ID AND YOUR JavaScript KEY
    Parse.serverURL = "https://parseapi.back4app.com/";

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
            emailBody += "\n Email: " + $('#email').val();
            const Survey = Parse.Object.extend("Survey");
            const survey = new Survey();
            survey.set("Email", emailBody);
            survey.save()
                .then((survey) => {
                // Execute any logic that should take place after the object is saved.
                alert('Thank you! Form was submitted successfully');
                }, (error) => {
                // Execute any logic that should take place if the save fails.
                // error is a Parse.Error with an error code and message.
                alert('Error occured submitting the form');
                });
            }
    })
})