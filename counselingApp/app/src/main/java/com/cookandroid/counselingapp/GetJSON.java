package com.cookandroid.counselingapp;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Iterator;

public class GetJSON {
     public String getJsonData(String inputJson){
          try
          {
               JSONObject jsonObject = new JSONObject(inputJson);
               String replyText = jsonObject.getString("Replytext");

               Log.d("GetJSON", "replytext 응답"+replyText);
               return replyText;
          }
          catch (JSONException e)
          {
               e.printStackTrace();
          }

          return null;
     }

}
