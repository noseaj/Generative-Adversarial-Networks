const express = require("express");
const fileUpload = require("express-fileupload");
const app = express();

app.set("view engine", "ejs");
app.use(express.urlencoded({ extended: true })); //body-parser 라이브러리 포함 대신
app.use(fileUpload());
app.use("/public", express.static("public"));

app.listen(8000, function () {
  console.log("8000포트로 접속하셨습니다.");
});

app.get("/", function (req, res) {
  res.render("index.ejs"); //렌더링 하려는 파일, ejs파일에 넣을 값 posts라는 이름으로
});

app.get("/generate/:id", function (req, res) {
  res.render("generate.ejs", { post: req.params.id });
});

app.get("/result", function (req, res) {
  res.render("result.ejs");
});

app.post("/upload", function (req, res) {
  let sampleFile;
  let uploadPath;
  if (!req.files || Object.keys(req.files).length === 0) {
    return res.status(400).send("No files were uploaded.");
  }
  sampleFile = req.files.sampleFile;
  uploadPath = __dirname + "/public/images/" + "test.jpg";
  sampleFile.mv(uploadPath, function (err) {
    if (err) return res.status(500).send(err);
    res.redirect("/generate/" + req.files.sampleFile.name);
  });
});

app.post("/transfer", function (req, res) {
  let { X, Y, imageName, leakHeight, leakWidth } = req.body;
  X = parseInt(X);
  Y = parseInt(Y);
  leakHeight = Math.abs(parseInt(leakHeight));
  leakWidth = Math.abs(parseInt(leakWidth));
  const spawn = require("child_process").spawn;
  if (isNaN(X) && isNaN(Y)) {
    const result_02 = spawn("python", ["genModel.py", imageName]);
    result_02.stdout.on("data", (result) => {
      res.redirect("/result");
    });
  } else {
    const result_02 = spawn("python", [
      "genModel2.py",
      Math.ceil((X * 5) / 2),
      Math.ceil((Y * 5) / 2),
      Math.ceil((leakWidth * 5) / 2),
      Math.ceil((leakHeight * 5) / 2),
    ]);
    result_02.stdout.on("data", (result) => {
      res.redirect("/result");
    });
  }
});
