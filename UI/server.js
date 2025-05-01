const express = require('express'); // Import the Express package (installed via NPM)
const fs = require('fs'); // Import the fs module (built-in, no need to install)
const { engine } = require('express-handlebars');
const app = express();







app.use(express.static('public'))
app.use(express.urlencoded({ extended: false }));
// Set the port for the server to listen on
const PORT = 8080;


app.engine('handlebars', engine({
    helpers: {
        eq(a, b) { return a == b; },
        eq(a, b, c, d) { return a == c && b == d },
    }
}))
// initialize the engine to be handlebars
app.set('view engine', 'handlebars') // set handlebars to be the view engine
app.set('views', './views') // define the views directory to be ./views






app.use((req, res, next) => {
    // Log the request method (GET or POST) and URL (/ or /login)
    console.log(req.method, req.url);
    next(); // Continue to the next middleware or route
});
//---



app.get('/', function (req, res) {
    //send a text massge back to the client

    res.render('home.handlebars')
})


app.get('/fine-tuning', function (req, res) {
    //send a text massge back to the client

    res.render('fine-tuning.handlebars')
})















app.listen(PORT, () => {
    //  creatproducts(db)
    //creatusers(db)
    // createmanytomany(db)

    console.log('http://localhost:8080');
});

