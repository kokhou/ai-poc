const fs = require('fs');
const readline = require('readline');
const axios = require('axios');

const fileStream = fs.createReadStream('extracted_user_questions.txt');
const rl = readline.createInterface({
  input: fileStream,
  crlfDelay: Infinity
});

rl.on('line', async (line) => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/post_endpoint', { data: line });
    console.log(`Line sent: ${line}, Response: ${response.data}`);
  } catch (error) {
    console.error(`Error sending line: ${line}, Error: ${error}`);
  }
});
