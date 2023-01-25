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
  const XArray = X.split(",");
  const YArray = Y.split(",");
  const heightArray = leakHeight.split(",");
  const widthArray = leakWidth.split(",");
  let leakNum = widthArray.length - 1;
  let leakInfo = [];
  for (var i = 0; i < leakNum; i++) {
    leakInfo.push([
      Math.ceil((parseInt(XArray.slice(1)[i]) * 5) / 2),
      Math.ceil((parseInt(YArray.slice(1)[i]) * 5) / 2),
      Math.ceil((parseInt(widthArray.slice(1)[i]) * 5) / 2),
      Math.ceil((parseInt(heightArray.slice(1)[i]) * 5) / 2),
    ]);
  }

  const spawn = require("child_process").spawn;

  if (leakHeight == 0) {
    const result_02 = spawn("python", ["genModel.py", imageName]);
    result_02.stdout.on("data", (result) => {
      res.redirect("/result");
    });
  } else {
    const result_02 = spawn("python", [
      "genModel2.py",
      leakNum,
      JSON.stringify(leakInfo),
    ]);
    result_02.stdout.on("data", (result) => {
      res.redirect("/result");
    });
  }
});
