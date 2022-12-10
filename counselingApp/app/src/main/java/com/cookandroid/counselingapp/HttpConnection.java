package com.cookandroid.counselingapp;

import android.os.AsyncTask;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class HttpConnection {
    String EXCEPTION_ERROR = "Exception Occured. Check the url";

    String sendMsg;
    String chatUrl;

    public String POSTFunction(String mUrl, String jsonInputString) {

        JSONObject postJson = new JSONObject();

        try {
            postJson.put("Usertext", jsonInputString);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        sendMsg = postJson.toString();
        chatUrl = mUrl;


        try {
            //받아온 String을 url로 만들어주기
            URL url = new URL(mUrl);

            //conn으로 url connection을 open 해주기
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            //불러오는데 시간이 오래 걸리는 경우 Time out 설정
            conn.setReadTimeout(10000);
            //연결하는데 시간이 오래 걸리는 경우 Time out 설정
            conn.setConnectTimeout(15000);
            //연결 방법 설정
            conn.setRequestMethod("POST");
            conn.setDoInput(true);
            conn.setDoOutput(true);

            //Accept-Charset 설정 UTF-8 or ASCII
            conn.setRequestProperty("Content-Type", "application/json; utf-8");
            conn.setRequestProperty("Accept", "application/json");

            try {

                // POST로 넘겨줄 파라미터 생성.
                byte[] outputInBytes = sendMsg.getBytes("UTF-8");
                OutputStream os = conn.getOutputStream();
                os.write(outputInBytes);
                os.close();
                Log.d("HttpConnection","서버 post 성공");


            }catch (Exception e){
                Log.d("HttpConnection","서버 post 실패");
            }



            //결과값을 받아온다.
            InputStream is = conn.getInputStream();
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String line;
            StringBuffer response = new StringBuffer();
            while ((line = br.readLine()) != null) {
                response.append(line);
                response.append('\r');
            }
            br.close();

            String res = response.toString();
            res = res.trim();

            Log.d("HttpConnection","서버 response"+res);


            return res;

        } catch (Exception e) {
            e.printStackTrace();
        }


        Log.d("ERROR", EXCEPTION_ERROR);
        return null;
    }
}

